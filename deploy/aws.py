"""Deploy the bot to AWS Lambda, using an AWS CLI profile (PROFILE / REGION below).

    uv run python deploy/aws.py deploy    # build + create/update everything (safe to re-run)
    uv run python deploy/aws.py migrate   # copy the computer bot's saved state to DynamoDB
    uv run python deploy/aws.py webhook   # point Telegram at Lambda (stop the computer bot first)
    uv run python deploy/aws.py unhook    # point Telegram away from Lambda (back to polling)
    uv run python deploy/aws.py status
    uv run python deploy/aws.py logs [hours]   # what the function did (default: last 24 hours)
    uv run python deploy/aws.py state          # slips being reviewed / saved / exported, per chat

Stays inside AWS's always-free limits: Lambda (1M requests/month), DynamoDB with 5/5
provisioned capacity (25 free), SSM standard parameters, EventBridge Scheduler, 14-day logs.
No VPC / NAT gateway (those cost money).
"""

from __future__ import annotations

import io
import json
import secrets
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
# Change the profile and region to your own.
PROFILE, REGION = "tally", "ap-southeast-2"
NAME = "tally-bot"
SSM_PREFIX = f"/{NAME}"
RUNTIME = "python3.12"
# Already in the Lambda runtime; leaving them out keeps the upload small.
IN_RUNTIME = ("boto3", "botocore", "s3transfer", "jmespath", "python-dateutil", "six", "urllib3")
# Non-secret settings copied from .env into the function's environment.
PLAIN_SETTINGS = ("ALLOWED_TELEGRAM_USER_IDS", "RATES_SHEET_URL", "LLM_MODEL", "LLM_PROVIDER",
                  "LLM_REASONING_EFFORT", "BATCH_WEEKDAY", "BATCH_TIME", "TIMEZONE")

session = boto3.Session(profile_name=PROFILE, region_name=REGION)
env = dotenv_values(ROOT / ".env")


def step(msg: str) -> None:
    print(f"• {msg}", flush=True)


def _exists(call, *codes) -> bool:
    try:
        call()
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in codes:
            return False
        raise


# -- build -------------------------------------------------------------------------

def build_zip() -> bytes:
    step("Building the Lambda package")
    shutil.rmtree(BUILD, ignore_errors=True)
    pkg = BUILD / "package"
    pkg.mkdir(parents=True)
    req = BUILD / "requirements.txt"
    subprocess.run(["uv", "export", "--no-dev", "--no-hashes", "--no-emit-project", "--quiet", "-o", str(req)],
                   check=True, cwd=ROOT)
    lines = [l for l in req.read_text().splitlines()
             if l.strip() and not l.startswith(("#", " ")) and l.split("==")[0].strip() not in IN_RUNTIME]
    req.write_text("\n".join(lines) + "\n")
    subprocess.run(["uv", "pip", "install", "--quiet", "--target", str(pkg), "--no-deps",
                    "--python-platform", "x86_64-manylinux_2_28", "--python-version", "3.12",
                    "--only-binary", ":all:", "-r", str(req)], check=True, cwd=ROOT)
    shutil.copytree(ROOT / "tally_agent", pkg / "tally_agent", ignore=shutil.ignore_patterns("__pycache__"))
    (pkg / "config").mkdir()
    for f in ("masters.toml", "hints.toml"):
        shutil.copy(ROOT / "config" / f, pkg / "config" / f)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(pkg.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                z.write(path, path.relative_to(pkg))
    data = buf.getvalue()
    step(f"Package: {len(data) / 1e6:.1f} MB zipped")
    if len(data) > 50e6:
        raise SystemExit("Package is over Lambda's 50 MB direct-upload limit.")
    return data


# -- pieces ----------------------------------------------------------------------------

def put_secrets() -> None:
    step("Storing secrets in SSM Parameter Store (encrypted)")
    ssm = session.client("ssm")
    values = {
        "TELEGRAM_BOT_TOKEN": env["TELEGRAM_BOT_TOKEN"],
        "OPENROUTER_API_KEY": env["OPENROUTER_API_KEY"],
        "GOOGLE_SERVICE_ACCOUNT_JSON": (ROOT / "config" / "google-service-account.json").read_text(),
    }
    try:
        ssm.get_parameter(Name=f"{SSM_PREFIX}/WEBHOOK_SECRET")
    except ssm.exceptions.ParameterNotFound:
        values["WEBHOOK_SECRET"] = secrets.token_urlsafe(32)
    for name, value in values.items():
        ssm.put_parameter(Name=f"{SSM_PREFIX}/{name}", Value=value, Type="SecureString",
                          Overwrite=True, Tier="Standard")


def ensure_table() -> str:
    ddb = session.client("dynamodb")
    if not _exists(lambda: ddb.describe_table(TableName=NAME), "ResourceNotFoundException"):
        # Pay per request: photos are ~300 KB (~300 write units each), so a few in a row would
        # exhaust a small provisioned allowance and get refused. On demand costs cents a month.
        step("Creating DynamoDB table (pay per request)")
        ddb.create_table(
            TableName=NAME, BillingMode="PAY_PER_REQUEST",
            AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"},
                                  {"AttributeName": "sk", "AttributeType": "S"}],
            KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}, {"AttributeName": "sk", "KeyType": "RANGE"}])
        ddb.get_waiter("table_exists").wait(TableName=NAME)
    return ddb.describe_table(TableName=NAME)["Table"]["TableArn"]


def ensure_role(account: str, table_arn: str) -> str:
    iam = session.client("iam")
    role = f"{NAME}-lambda"
    trust = {"Version": "2012-10-17", "Statement": [{
        "Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
    created = not _exists(lambda: iam.get_role(RoleName=role), "NoSuchEntity")
    if created:
        step("Creating the function's IAM role")
        iam.create_role(RoleName=role, AssumeRolePolicyDocument=json.dumps(trust))
        iam.attach_role_policy(RoleName=role,
                               PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole")
    fn_arn = f"arn:aws:lambda:{REGION}:{account}:function:{NAME}"
    iam.put_role_policy(RoleName=role, PolicyName=f"{NAME}-access", PolicyDocument=json.dumps({
        "Version": "2012-10-17", "Statement": [
            {"Effect": "Allow", "Action": ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:DeleteItem",
                                           "dynamodb:Query"], "Resource": table_arn},
            {"Effect": "Allow", "Action": "ssm:GetParameters",
             "Resource": f"arn:aws:ssm:{REGION}:{account}:parameter{SSM_PREFIX}/*"},
            {"Effect": "Allow", "Action": "kms:Decrypt", "Resource": "*",
             "Condition": {"StringEquals": {"kms:ViaService": f"ssm.{REGION}.amazonaws.com"}}},
            {"Effect": "Allow", "Action": "lambda:InvokeFunction", "Resource": fn_arn},
        ]}))
    if created:
        time.sleep(10)   # new IAM roles take a moment before Lambda can use them
    return iam.get_role(RoleName=role)["Role"]["Arn"]


def ensure_function(role_arn: str, code: bytes) -> str:
    lam = session.client("lambda")
    variables = {"STORE": "dynamodb", "DYNAMODB_TABLE": NAME, "SSM_PREFIX": SSM_PREFIX}
    variables |= {k: env[k] for k in PLAIN_SETTINGS if env.get(k)}
    config = dict(Role=role_arn, Runtime=RUNTIME, Handler="tally_agent.lambda_app.handler",
                  MemorySize=512, Timeout=300, Environment={"Variables": variables},
                  Architectures=["x86_64"])
    if _exists(lambda: lam.get_function(FunctionName=NAME), "ResourceNotFoundException"):
        step("Updating the function code and settings")
        lam.update_function_code(FunctionName=NAME, ZipFile=code)
        lam.get_waiter("function_updated_v2").wait(FunctionName=NAME)
        config.pop("Architectures")
        lam.update_function_configuration(FunctionName=NAME, **config)
    else:
        step("Creating the function")
        for attempt in range(6):
            try:
                lam.create_function(FunctionName=NAME, Code={"ZipFile": code}, **config)
                break
            except ClientError as e:
                if "cannot be assumed" not in str(e) or attempt == 5:
                    raise
                time.sleep(5)
    lam.get_waiter("function_updated_v2").wait(FunctionName=NAME)
    # A failed background run must not be retried: it may already have messaged the user.
    lam.put_function_event_invoke_config(FunctionName=NAME, MaximumRetryAttempts=0,
                                         MaximumEventAgeInSeconds=3600)
    return lam.get_function(FunctionName=NAME)["Configuration"]["FunctionArn"]


def ensure_logs() -> None:
    logs = session.client("logs")
    group = f"/aws/lambda/{NAME}"
    if not _exists(lambda: logs.create_log_group(logGroupName=group), "ResourceAlreadyExistsException"):
        pass
    logs.put_retention_policy(logGroupName=group, retentionInDays=14)


def ensure_url() -> str:
    lam = session.client("lambda")
    try:
        url = lam.get_function_url_config(FunctionName=NAME)["FunctionUrl"]
    except lam.exceptions.ResourceNotFoundException:
        step("Creating the function URL (Telegram calls this; requests need the webhook secret)")
        url = lam.create_function_url_config(FunctionName=NAME, AuthType="NONE")["FunctionUrl"]
    for sid, kwargs in (("public-url", dict(Action="lambda:InvokeFunctionUrl", FunctionUrlAuthType="NONE")),
                        ("public-url-invoke", dict(Action="lambda:InvokeFunction", InvokedViaFunctionUrl=True))):
        try:
            lam.add_permission(FunctionName=NAME, StatementId=sid, Principal="*", **kwargs)
        except lam.exceptions.ResourceConflictException:
            pass
    return url


def ensure_schedule(account: str, fn_arn: str) -> None:
    iam = session.client("iam")
    role = f"{NAME}-scheduler"
    if not _exists(lambda: iam.get_role(RoleName=role), "NoSuchEntity"):
        step("Creating the scheduler's IAM role")
        iam.create_role(RoleName=role, AssumeRolePolicyDocument=json.dumps({
            "Version": "2012-10-17", "Statement": [{"Effect": "Allow",
                                                    "Principal": {"Service": "scheduler.amazonaws.com"},
                                                    "Action": "sts:AssumeRole"}]}))
        time.sleep(10)
    iam.put_role_policy(RoleName=role, PolicyName="invoke", PolicyDocument=json.dumps({
        "Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Action": "lambda:InvokeFunction",
                                               "Resource": fn_arn}]}))
    role_arn = iam.get_role(RoleName=role)["Role"]["Arn"]

    weekday = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"][int(env.get("BATCH_WEEKDAY") or 5)]
    hour, minute = (env.get("BATCH_TIME") or "21:00").split(":")
    schedule = dict(Name=f"{NAME}-weekly", ScheduleExpression=f"cron({int(minute)} {int(hour)} ? * {weekday} *)",
                    ScheduleExpressionTimezone=env.get("TIMEZONE") or "Asia/Kolkata",
                    FlexibleTimeWindow={"Mode": "OFF"},
                    Target={"Arn": fn_arn, "RoleArn": role_arn, "Input": json.dumps({"tally": "weekly"}),
                            "RetryPolicy": {"MaximumRetryAttempts": 2}})
    sch = session.client("scheduler")
    if _exists(lambda: sch.get_schedule(Name=schedule["Name"]), "ResourceNotFoundException"):
        sch.update_schedule(**schedule)
    else:
        step(f"Creating the weekly schedule: {schedule['ScheduleExpression']} {schedule['ScheduleExpressionTimezone']}")
        sch.create_schedule(**schedule)


# -- commands ------------------------------------------------------------------------

def deploy() -> None:
    account = session.client("sts").get_caller_identity()["Account"]
    step(f"AWS account {account}, region {REGION}, profile {PROFILE}")
    code = build_zip()
    put_secrets()
    table_arn = ensure_table()
    role_arn = ensure_role(account, table_arn)
    fn_arn = ensure_function(role_arn, code)
    ensure_logs()
    url = ensure_url()
    ensure_schedule(account, fn_arn)
    step(f"Done. Function URL: {url}")
    step("Telegram still talks to the computer bot until you run: deploy/aws.py webhook")


def telegram(method: str, **params) -> dict:
    req = urllib.request.Request(f"https://api.telegram.org/bot{env['TELEGRAM_BOT_TOKEN']}/{method}",
                                 data=json.dumps(params).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def webhook() -> None:
    url = session.client("lambda").get_function_url_config(FunctionName=NAME)["FunctionUrl"]
    secret = session.client("ssm").get_parameter(Name=f"{SSM_PREFIX}/WEBHOOK_SECRET",
                                                 WithDecryption=True)["Parameter"]["Value"]
    result = telegram("setWebhook", url=url, secret_token=secret, drop_pending_updates=False,
                      allowed_updates=["message", "callback_query"])
    print(result)


def unhook() -> None:
    print(telegram("deleteWebhook", drop_pending_updates=False))


def status() -> None:
    info = telegram("getWebhookInfo")["result"]
    print(json.dumps({k: info.get(k) for k in ("url", "pending_update_count", "last_error_message",
                                               "last_error_date")}, indent=1))


def logs(hours: str = "24") -> None:
    """The function's log lines, oldest first, without the per-call START/END noise."""
    import datetime

    client = session.client("logs")
    kw = {"logGroupName": f"/aws/lambda/{NAME}", "startTime": int((time.time() - float(hours) * 3600) * 1000)}
    while True:
        page = client.filter_log_events(**kw)
        for e in page["events"]:
            msg = e["message"].rstrip()
            if msg.startswith(("START ", "END ")) or "Found credentials" in msg:
                continue
            when = datetime.datetime.fromtimestamp(e["timestamp"] / 1000).strftime("%m-%d %H:%M:%S")
            print(when, msg[:400])
        if "nextToken" not in page:
            break
        kw["nextToken"] = page["nextToken"]


def state() -> None:
    """Who has what in the bot right now (names come from the .env user list order)."""
    table = session.resource("dynamodb").Table(NAME)
    users = env.get("ALLOWED_TELEGRAM_USER_IDS", "").split(",")
    rows, kw = [], {}
    while True:
        page = table.scan(**kw)
        rows += page["Items"]
        if "LastEvaluatedKey" not in page:
            break
        kw["ExclusiveStartKey"] = page["LastEvaluatedKey"]

    def heading(slip: dict) -> str:
        return f"{slip['party'].strip()} · {slip['date']} · challan {slip['challan_number']} · {len(slip['lines'])} lines"

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["pk"]] = counts.get(r["pk"], 0) + 1
        data = json.loads(r["data"]) if "data" in r else {}
        chat = str(data.get("chat_id", ""))
        who = f"user {users.index(chat) + 1}" if chat in users else chat
        if r["pk"] == "draft":
            print(f"reviewing [{who}]  {heading(data['slip'])}")
        elif r["pk"] == "pending":
            print(f"saved     [{who}]  {heading(data['slip'])}  (saved {data['saved_at']})")
        elif r["pk"] == "exported":
            print(f"exported           {r['sk'].strip()}  ({data['exported_at']}, {data['export']})")
    print("records:", dict(sorted(counts.items())))


def migrate() -> None:
    """Copy the computer bot's saved state (data/, as written by the polling version) into
    DynamoDB: slips being reviewed, slips saved for Saturday, learned spellings, open rate
    questions, the export index and the weekly-run marker. Run after stopping the computer bot."""
    import pickle

    sys.path.insert(0, str(ROOT))
    from tally_agent.kv import DynamoKV
    from tally_agent.models import Slip
    from tally_agent.store import slip_key

    data = ROOT / "data"
    dst = DynamoKV.__new__(DynamoKV)
    dst.table = session.resource("dynamodb").Table(NAME)
    count = 0

    def put(ns, key, value):
        nonlocal count
        dst.put(ns, key, value)
        count += 1

    if (data / "batch_state.json").exists():
        put("state", "batch", json.loads((data / "batch_state.json").read_text()))
    if (data / "learned_aliases.json").exists():
        for key, row in json.loads((data / "learned_aliases.json").read_text()).items():
            put("learned", key, row)
    for f in sorted((data / "pending").glob("*.json")):
        put("pending", f.stem, json.loads(f.read_text()))
    for f in sorted((data / "inbox").glob("*.jpg")):
        from tally_agent.store import _fit_jpeg
        dst.put_bytes("photo", f.stem, _fit_jpeg(f.read_bytes()))
        count += 1
    index = data / "exports" / "index.jsonl"
    for row in map(json.loads, index.read_text().splitlines() if index.exists() else []):
        put("exported", row["key"], row)

    state = data / "bot_state.pickle"
    chats = pickle.loads(state.read_bytes()).get("chat_data", {}) if state.exists() else {}
    for chat_id, cd in chats.items():
        for draft_id, d in cd.get("drafts", {}).items():
            Slip.model_validate(d["slip"])   # fail loudly on an unexpected shape
            put("draft", draft_id, {"slip": d["slip"], "chat_id": chat_id,
                                    "preview_message_id": d.get("preview_message_id")})
            if d.get("preview_message_id"):
                put("msg", f"{chat_id}:{d['preview_message_id']}", {"kind": "draft", "draft_id": draft_id})
        if cd.get("last_draft"):
            put("last_draft", str(chat_id), {"draft_id": cd["last_draft"]})
        for message_id, (party, item) in (cd.get("rate_questions") or {}).items():
            put("msg", f"{chat_id}:{message_id}", {"kind": "rate_q", "party": party, "item": item})
        for draft_id, updates in (cd.get("rate_updates") or {}).items():
            put("rate_update", draft_id, {"updates": [list(u) for u in updates]})
    step(f"Copied {count} records to DynamoDB")


if __name__ == "__main__":
    commands = {"deploy": deploy, "migrate": migrate, "webhook": webhook, "unhook": unhook, "status": status,
                "logs": logs, "state": state}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        raise SystemExit(__doc__)
    commands[sys.argv[1]](*sys.argv[2:])
