#!/bin/bash
# gbar_run.sh -- run gbar batch operations from your laptop without typing
# the DTU password more than once.
#
# How it works:
#   1. Prompts ONCE for DTU password using `read -s` (silent, no echo).
#   2. Stores it briefly in an env var, fed to `expect` to drive ssh.
#   3. SSH ControlMaster (set in ~/.ssh/config) keeps the auth session
#      alive ~30 min so subsequent invocations need no password.
#
# Where the password lives:
#   - shell variable in this process (cleared at exit via `unset`)
#   - briefly exported to an `expect` child process (cleared at exit)
#   - never on disk, never in `ps` (set, not on cmdline), never in logs
#
# Usage:
#   ./gbar_run.sh deploy      # clone/update repo + submit phase2 PPO job
#   ./gbar_run.sh status      # bjobs + tail logs
#   ./gbar_run.sh fetch       # rsync results back to local
#   ./gbar_run.sh shell       # interactive ssh

set -euo pipefail

PASSPHRASE="cats"   # ssh key passphrase (from ~/.gbar.md, low-sensitivity)
REMOTE_DIR='~/projects/battery_gym'
REPO_URL="https://github.com/kilojoules/battery_gym.git"
LOCAL_RESULTS_DIR="./gbar_results"

cmd="${1:-status}"

prompt_password() {
    if [[ -n "${DTU_PW:-}" ]]; then return; fi
    # Skip prompt if ControlMaster socket exists -- session is alive
    local sock="$HOME/.ssh/cm-juqu@login1.gbar.dtu.dk:22"
    if [[ -S "$sock" ]] && ssh -O check gbar 2>/dev/null; then
        echo "[gbar_run] ControlMaster session alive, no password needed."
        export DTU_PW=""
        return
    fi
    if [[ ! -t 0 ]]; then
        echo "[gbar_run] No TTY and no ControlMaster session. Please re-run from interactive shell." >&2
        exit 4
    fi
    echo "[gbar_run] DTU password (off-VPN 2FA) -- input hidden:"
    read -s DTU_PW
    echo
    export DTU_PW
}

# --- expect-driven ssh ------------------------------------------------------
ssh_run() {
    local remote_cmd="$1"
    DTU_PW="${DTU_PW:-}" PASSPHRASE_ARG="$PASSPHRASE" REMOTE="$remote_cmd" \
        expect <<'EXPECT_EOF'
set timeout 600
set passphrase $env(PASSPHRASE_ARG)
set dtupw $env(DTU_PW)
set remote $env(REMOTE)
log_user 1
spawn ssh -o StrictHostKeyChecking=accept-new gbar $remote
expect {
    -re "Enter passphrase for key.*:" {
        send "$passphrase\r"
        exp_continue
    }
    -re "(P|p)assword:" {
        if {[string length $dtupw] == 0} {
            puts stderr "\n\[gbar_run\] DTU password prompted but DTU_PW empty. Aborting."
            exit 2
        }
        send "$dtupw\r"
        exp_continue
    }
    "denied" {
        puts stderr "\n\[gbar_run\] auth denied -- check passphrase / DTU password"
        exit 3
    }
    eof
}
catch wait result
exit [lindex $result 3]
EXPECT_EOF
}

ssh_rsync() {
    local from="$1"; local to="$2"
    DTU_PW="${DTU_PW:-}" PASSPHRASE_ARG="$PASSPHRASE" SRC="$from" DST="$to" \
        expect <<'EXPECT_EOF'
set timeout 600
set passphrase $env(PASSPHRASE_ARG)
set dtupw $env(DTU_PW)
set src $env(SRC)
set dst $env(DST)
log_user 1
spawn rsync -avz --progress -e "ssh" $src $dst
expect {
    -re "Enter passphrase for key.*:" { send "$passphrase\r"; exp_continue }
    -re "(P|p)assword:" {
        if {[string length $dtupw] == 0} {
            puts stderr "\nDTU_PW empty"; exit 2
        }
        send "$dtupw\r"; exp_continue
    }
    eof
}
catch wait result
exit [lindex $result 3]
EXPECT_EOF
}

cleanup() {
    unset DTU_PW PASSPHRASE_ARG REMOTE SRC DST 2>/dev/null || true
}
trap cleanup EXIT

case "$cmd" in
    deploy)
        prompt_password
        ssh_run "set -e
            source /etc/profile 2>/dev/null || true
            [ -f ~/.bashrc ] && source ~/.bashrc 2>/dev/null || true
            command -v bsub >/dev/null || PATH=/usr/local/lsf/bin:/opt/ibm/lsfsuite/lsf/10.1/linux2.6-glibc2.3-x86_64/bin:\$PATH
            command -v bsub >/dev/null || { echo bsub not found in PATH; exit 1; }
            mkdir -p ~/projects
            cd ~/projects
            if [ -d battery_gym/.git ]; then
                cd battery_gym && git fetch && git reset --hard origin/main
            else
                rm -rf battery_gym
                git clone $REPO_URL battery_gym
                cd battery_gym
            fi
            chmod +x gbar_phase2_ppo.sh
            bsub < gbar_phase2_ppo.sh"
        ;;
    status)
        prompt_password
        ssh_run "source /etc/profile 2>/dev/null; [ -f ~/.bashrc ] && source ~/.bashrc 2>/dev/null
            command -v bjobs >/dev/null || PATH=/usr/local/lsf/bin:/opt/ibm/lsfsuite/lsf/10.1/linux2.6-glibc2.3-x86_64/bin:\$PATH
            echo === bjobs === ; bjobs -a -W 2>&1 | head -20 ; echo ; echo === out tail === ; tail -40 $REMOTE_DIR/phase2_ppo_*.out 2>/dev/null | tail -40 ; echo ; echo === err tail === ; tail -20 $REMOTE_DIR/phase2_ppo_*.err 2>/dev/null | tail -20"
        ;;
    fetch)
        prompt_password
        mkdir -p "$LOCAL_RESULTS_DIR"
        ssh_rsync "gbar:$REMOTE_DIR/ppo_logs/" "$LOCAL_RESULTS_DIR/"
        ssh_rsync "gbar:$REMOTE_DIR/ppo_policy.zip" "$LOCAL_RESULTS_DIR/"
        ssh_rsync "gbar:$REMOTE_DIR/phase2_ppo_*.out" "$LOCAL_RESULTS_DIR/"
        echo "[gbar_run] results in $LOCAL_RESULTS_DIR/"
        ;;
    shell)
        prompt_password
        # Interactive shell -- expect needs interact mode
        DTU_PW="${DTU_PW:-}" PASSPHRASE_ARG="$PASSPHRASE" expect <<'EXPECT_EOF'
set timeout 60
set passphrase $env(PASSPHRASE_ARG)
set dtupw $env(DTU_PW)
spawn ssh -o StrictHostKeyChecking=accept-new gbar
expect {
    -re "Enter passphrase for key.*:" { send "$passphrase\r"; exp_continue }
    -re "(P|p)assword:" {
        if {[string length $dtupw] == 0} { interact; exit }
        send "$dtupw\r"; exp_continue
    }
    -re "\\$ |# " { interact }
}
EXPECT_EOF
        ;;
    cancel-master)
        ssh -O exit gbar 2>/dev/null || true
        echo "[gbar_run] ControlMaster session closed."
        ;;
    wait-and-fetch)
        # Poll bjobs until DONE/EXIT, then fetch + plot. ControlMaster keeps
        # us authenticated across polls (no re-prompt within 30 min idle).
        prompt_password
        echo "[gbar_run] polling job status every 60 s..."
        # Disable strict mode in this branch -- ssh_run returning non-zero
        # transiently shouldn't kill the whole watcher.
        set +e
        while true; do
            STATUS=$(ssh_run 'source /etc/profile 2>/dev/null; bjobs -a -noheader 2>&1 | head -1 | awk "{print \$6}"' 2>&1 | tail -1)
            echo "  $(date +%H:%M:%S)  status=$STATUS"
            case "$STATUS" in
                DONE|EXIT) break ;;
                RUN|PEND|PSUSP|USUSP|SSUSP|WAIT|UNKWN) sleep 60 ;;
                "") echo "  no jobs found, exiting" ; exit 0 ;;
                *) echo "  unrecognized status [$STATUS], sleeping 60" ; sleep 60 ;;
            esac
        done
        set -e
        mkdir -p "$LOCAL_RESULTS_DIR"
        ssh_rsync "gbar:$REMOTE_DIR/ppo_logs/" "$LOCAL_RESULTS_DIR/"
        ssh_rsync "gbar:$REMOTE_DIR/ppo_policy.zip" "$LOCAL_RESULTS_DIR/" || true
        ssh_rsync "gbar:$REMOTE_DIR/phase2_ppo_*.out" "$LOCAL_RESULTS_DIR/" || true
        echo "[gbar_run] fetched. running plot..."
        python3 phase2_ppo_plot.py "$LOCAL_RESULTS_DIR/ppo_logs/phase2_ppo_results.json"
        ;;
    diag)
        prompt_password
        ssh_run "echo === PATH === ; echo \$PATH ; echo ; echo === bsub === ; command -v bsub ; echo ; echo === bashrc head === ; head -30 ~/.bashrc 2>/dev/null ; echo ; echo === LSF locations === ; ls -d /opt/ibm/lsfsuite 2>/dev/null ; ls -d /usr/local/lsf 2>/dev/null ; ls /etc/profile.d/ 2>/dev/null | grep -i lsf ; echo === after profile === ; source /etc/profile 2>/dev/null ; source ~/.bashrc 2>/dev/null ; echo PATH=\$PATH ; command -v bsub"
        ;;
    *)
        echo "Usage: $0 {deploy|status|fetch|shell|cancel-master|diag|wait-and-fetch}"
        exit 1
        ;;
esac

echo "[gbar_run] $cmd done."
