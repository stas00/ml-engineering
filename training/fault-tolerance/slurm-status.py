#!/usr/bin/env python

#
# This tool reports on the status of the job - whether it's running or scheduled and various other
# useful data
#
# Example:
#
# slurm-status.py --job-name tr1-13B-round3
#

import argparse
import io
import json
import os
import re
import shlex
import smtplib
import socket
import subprocess
import sys
from datetime import datetime, timedelta

SLURM_GROUP_NAME = "six"

# this needs to be an actual email subscribed to bigscience@groups.google.com
FROM_ADDR = "bigscience-bot@huggingface.co"
TO_ADDRS = ["bigscience@googlegroups.com", "foo@bar.com"] # wants a list

def send_email(subject, body):
    message = f"""\
From: {FROM_ADDR}
To: {", ".join(TO_ADDRS)}
Subject: {subject}

{body}
"""

    server = smtplib.SMTP("localhost")
    #server.set_debuglevel(3)  # uncomment if need to debug
    server.sendmail(FROM_ADDR, TO_ADDRS, message)
    server.quit()

def send_email_alert_job_not_scheduled(job_name):

    subject = f"[ALERT] {job_name} is neither running nor scheduled to run"
    body = f"""
***ALERT: {job_name} is neither RUNNING nor SCHEDULED! Alert someone at Eng WG***

Please reply to this email once the issue has been taken care of, or if you are in the process of doing that, should new alerts be sent again.

If unsure what to do, please post in the #bigscience-engineering slack channel.

*** Useful info ***

On call info: https://github.com/bigscience-workshop/bigscience/tree/master/train/tr1-13B-base#on-call
Training logs: https://github.com/bigscience-workshop/bigscience/tree/master/train/tr1-13B-base#watching-the-training-logs
Launching training: https://github.com/bigscience-workshop/bigscience/tree/master/train/tr1-13B-base#training-scripts
"""

    send_email(subject, body)

def check_running_on_jean_zay():
    fqdn = socket.getfqdn()
    # sometimes it gives fqdn, other times it doesn't, so try to use both patterns
    if not ("idris.fr" in fqdn or "idrsrv" in fqdn):
        raise ValueError("This script relies on JZ's specific environment and won't work elsewhere. "
        f"You're attempting to run it on '{fqdn}'.")

def run_cmd(cmd):
    try:
        git_status = subprocess.run(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            check=True,
            encoding="utf-8",
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise EnvironmentError(exc.stderr)

    return git_status


def get_slurm_group_status():
    # we need to monitor slurm jobs of the whole group six, since the slurm job could be owned by
    # any user in that group
    cmd = f"getent group {SLURM_GROUP_NAME}"
    getent = run_cmd(cmd.split())
    # sample output: six:*:3015222:foo,bar,tar
    usernames = getent.split(':')[-1]

    # get all the scheduled and running jobs
    # use shlex to split correctly and not on whitespace
    cmd = f'squeue --user={usernames} -o "%.16i %.9P %.40j %.8T %.10M %.6D %.20S %R"'
    data = run_cmd(shlex.split(cmd))
    lines = [line.strip() for line in data.split("\n")]
    return lines


def get_remaining_time(time_str):
    """
    slurm style time_str = "2021-08-06T15:23:46"
    """

    delta = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S") - datetime.now()
    # round micsecs
    delta -= timedelta(microseconds=delta.microseconds)
    return delta


def get_preamble():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # add a string that is easy to grep for:
    return f"[{timestamp}] PULSE:"


def process_job(jobid, partition, name, state, time, nodes, start_time, notes):

    job_on_partition = f"{jobid} on '{partition}' partition"
    preamble = get_preamble()

    if state == "RUNNING":
        print(f"{preamble} {name} is running for {time} since {start_time} ({job_on_partition} ({notes})")
    elif state == "PENDING":
        if start_time == "N/A":
            if notes == "(JobArrayTaskLimit)":
                print(f"{preamble} {name} is waiting for the previous Job Array job to finish before scheduling a new one ({job_on_partition})")
            elif notes == "(Dependency)":
                print(f"{preamble} {name} is waiting for the previous job to finish before scheduling a new one using the dependency mechanism ({job_on_partition})")
            else:
                print(f"{preamble} {name} is waiting to be scheduled ({job_on_partition})")
        else:
            remaining_wait_time = get_remaining_time(start_time)
            print(f"{preamble} {name} is scheduled to start in {remaining_wait_time} (at {start_time}) ({job_on_partition})")

        return True
    else:
        # Check that we don't get some 3rd state
        print(f"{preamble} {name} is unknown - fix me: (at {start_time}) ({job_on_partition}) ({notes})")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name", type=str, required=True, help="slurm job name")
    parser.add_argument("-d", "--debug", action='store_true', help="enable debug")
    parser.add_argument("--no-email", action='store_true', help="do not email alerts")
    return parser.parse_args()


def main():

    check_running_on_jean_zay()

    args = get_args()
    status_lines = get_slurm_group_status()

    in_the_system = False
    for l in status_lines:
        #print(f"l=[{l}]")

        # XXX: apparently some jobs can be run w/o name and break the split() call, so match our
        # name first and then split
        if args.job_name in l:
            jobid, partition, name, state, time, nodes, start_time, notes = l.split(None, 7)
            #print("-".join([jobid, partition, name, state, time, nodes, start_time, notes]))
            # XXX: add support for regex matching so partial name can be provided
            if name == args.job_name:
                in_the_system = True
                process_job(jobid, partition, name, state, time, nodes, start_time, notes)

    if not in_the_system:
        preamble = get_preamble()
        print(f"{preamble} ***ALERT: {args.job_name} is not RUNNING or SCHEDULED! Alert someone at Eng WG***")
        if not args.no_email:
            send_email_alert_job_not_scheduled(args.job_name)


if __name__ == "__main__":

    main()
