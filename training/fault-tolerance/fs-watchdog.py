#!/usr/bin/env python

#
# This tool alerts on the status of the filesystem - when it's getting close to running out of disk space or inodes on various partitions at JZ
#
# Example:
#
# fs-watchdog.py
#

import argparse
import re
import smtplib
import socket
import subprocess
import sys

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

def send_email_alert(msg):

    subject = f"[ALERT] JZ filesystem is getting close to being full"
    body = f"""
***ALERT: One or more partitions at JZ are getting close to being full! Alert someone at Eng WG***

{msg}

Please reply to this email once the issue has been taken care of, or if you are in the process of doing that, should new alerts be sent again.

If unsure what to do, please post in the #bigscience-engineering slack channel.

"""

    send_email(subject, body)

def check_running_on_jean_zay():
    fqdn = socket.getfqdn()
    # sometimes it gives fqdn, other times it doesn't, so try to use both patterns
    if not ("idris.fr" in fqdn or "idrsrv" in fqdn):
        raise ValueError("This script relies on JZ's specific environment and won't work elsewhere. "
        f"You're attempting to run it on '{fqdn}'.")

def run_cmd(cmd, check=True):
    try:
        git_status = subprocess.run(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            check=check,
            encoding="utf-8",
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise EnvironmentError(exc.stderr)

    return git_status


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action='store_true', help="enable debug")
    parser.add_argument("--no-email", action='store_true', help="do not email alerts")
    return parser.parse_args()

def main():

    check_running_on_jean_zay()
    args = get_args()

    alerts = []
    def analyse_partition_bytes(partition_name, partition_path, hard_limit_bytes, alert_bytes_threshold):
        soft_limit_bytes = hard_limit_bytes * alert_bytes_threshold
        cmd = f"du -bs {partition_path}"
        response = run_cmd(cmd.split(), check=False) # du could report partial errors for wrong perms
        size_bytes = int(response.split()[0])
        if args.debug:
            print(f"{partition_name} bytes: {size_bytes}")

        if size_bytes > soft_limit_bytes:
            current_usage_percent = 100*size_bytes/hard_limit_bytes
            alerts.append(f"{partition_name} is at {current_usage_percent:.2f}% bytes usage ({size_bytes/2**30:.2f}GB/{hard_limit_bytes/2**30:.2f}GB)")
            alerts.append("")

    def analyse_partition_inodes(partition_name, partition_path, hard_limit_inodes, alert_inodes_threshold):
        soft_limit_inodes = hard_limit_inodes * alert_inodes_threshold
        cmd = f"du -s -BK --inodes {partition_path}"
        response = run_cmd(cmd.split(), check=False) # du could report partial errors for wrong perms
        size_inodes = int(response.split()[0])
        if args.debug:
            print(f"{partition_name} Inodes: {size_inodes}")

        if size_inodes > soft_limit_inodes:
            current_usage_percent = 100*size_inodes/hard_limit_inodes
            alerts.append(f"{partition_name} is at {current_usage_percent:.2f}% inodes usage ({size_inodes/2**10:.2f}K/{hard_limit_inodes/2**10:.2f}K)")
            alerts.append("")

    def analyse_partition_idrquota(partition_name, partition_flag, alert_bytes_threshold, alert_inodes_threshold):
        cmd = f"idrquota {partition_flag} -p {SLURM_GROUP_NAME}"
        response = run_cmd(cmd.split())
        match = re.findall(' \(([\d\.]+)%\)', response)
        if match:
            bytes_percent, inodes_percent = [float(x) for x in match]
        else:
            raise ValueError(f"{cmd} failed")
        if args.debug:
            print(f"{partition_name} bytes: {bytes_percent}%")
            print(f"{partition_name} inodes: {inodes_percent}%")

        msg = []
        if bytes_percent/100 > alert_bytes_threshold:
            msg.append(f"{partition_name} is at {bytes_percent:.2f}% bytes usage")

        if inodes_percent/100 > alert_inodes_threshold:
            msg.append(f"{partition_name} is at {inodes_percent:.2f}% inodes usage")

        if len(msg) > 0:
            alerts.extend(msg)
            alerts.append(response)
            alerts.append("")

    def analyse_shared_disk(partition_name, alert_bytes_threshold):
        partition_name_2_disk = {
            "SCRATCH": "gpfsssd",
            "WORK": "gpfsdswork",
            "STORE": "gpfsdsstore"
        }
        cmd = "df"
        response = run_cmd(cmd.split())
        disk_metas = response.split("\n")
        column_names = disk_metas[0].split()
        disk_meta = [disk_meta_.split() for disk_meta_ in disk_metas if disk_meta_.startswith(partition_name_2_disk[partition_name])][0]
        disk_meta = {column_name: value for column_name, value in zip(column_names, disk_meta)}

        # default `df` counts uses 1024-byte units, and `1024 == 2 ** 10`
        available_disk_left = int(disk_meta["Available"]) * 2 ** 10
        if available_disk_left < alert_bytes_threshold:
            alerts.append(f"Shared {partition_name} has {available_disk_left/2**40:.2f}TB left")
            alerts.append("")

    # WORK and STORE partitions stats can be accessed much faster through `idrquota`, and it already
    # includes the quota info
    analyse_partition_idrquota(partition_name="WORK", partition_flag="-w", alert_bytes_threshold=0.85, alert_inodes_threshold=0.85)
    analyse_partition_idrquota(partition_name="STORE", partition_flag="-s", alert_bytes_threshold=0.85, alert_inodes_threshold=0.85)

    # SCRATCH - check only bytes w/ a hard quota of 400TB - alert on lower threshold than other
    # partitions due to it filling up at a faster rate (dumping huge checkpoints)
    analyse_partition_bytes(partition_name="SCRATCH", partition_path="/gpfsssd/scratch/rech/six/", hard_limit_bytes=400*2**40, alert_bytes_threshold=0.75)
    # Actually SCRATCH is shared with everyone and we should monitor the output of `df -h | grep gpfsssd`
    # Check that there's still 40TB left
    analyse_shared_disk("SCRATCH", 100 * 2 ** 40)

    # WORKSF - check both bytes and inodes w/ hard quotas of 2TB / 3M
    analyse_partition_bytes(partition_name="WORKSF", partition_path="/gpfsssd/worksf/projects/rech/six/", hard_limit_bytes=2*2**40, alert_bytes_threshold=0.85)
    analyse_partition_inodes(partition_name="WORKSF", partition_path="/gpfsssd/worksf/projects/rech/six/", hard_limit_inodes=3*10**6, alert_inodes_threshold=0.85)

    if len(alerts) > 0 :
        print(f"[ALERT] JZ filesystem is getting close to being full")
        msg = "\n".join(alerts)
        print(msg)

        if not args.no_email:
            send_email_alert(msg)
    else:
        print("All partitions are in a good standing")

if __name__ == "__main__":

    main()
