#!/usr/bin/env bash
# Stop this RunPod pod when training ends (successfully or not), after saving results to the
# persistent volume. A budget cap stops it after MAX_MINUTES no matter what.
#   usage (on the pod): nohup bash scripts/runpod_finish.sh <train-pid> <run-name> [max-minutes] [grace-minutes] &
set -u
PID=$1; RUN=$2; MAX_MINUTES=${3:-570}; GRACE_MIN=${4:-30}
SRC=/root/OrchestraMaker/checkpoints/$RUN
DST=/workspace/OrchestraMaker/checkpoints/$RUN
DEADLINE=$(( $(date +%s) + MAX_MINUTES * 60 ))

log() { echo "$(date '+%F %T') $*"; }
save() { mkdir -p "$DST"; cp -f "$SRC"/{best.pt,last.pt,train.log,config.json} "$DST"/ 2>/dev/null; log "saved to $DST: $(ls "$DST" | tr '\n' ' ')"; }

log "watching training pid $PID, budget cap $MAX_MINUTES min"
while kill -0 "$PID" 2>/dev/null; do
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
        log "budget cap reached: stopping training"
        kill "$PID"; sleep 60
        break
    fi
    sleep 60
done
log "training process ended: $(tail -1 "$SRC/train.log")"
save
log "grace period $GRACE_MIN min for the local mirror to fetch the final checkpoint"
sleep $(( GRACE_MIN * 60 ))
save
log "stopping pod $RUNPOD_POD_ID"
runpodctl stop pod "$RUNPOD_POD_ID"
