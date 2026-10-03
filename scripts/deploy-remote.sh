#!/usr/bin/env bash
# Builds a release from an uploaded source snapshot and installs it under
# ~/services/lilia. Runs as the service user; never needs sudo.
set -euo pipefail

incoming_arg=${1:?usage: deploy-remote.sh <incoming> <commit> <dirty> <restart>}
commit=${2:?}
dirty=${3:?}
restart=${4:?}

root="$HOME/services/lilia"
incoming=$(realpath -m -- "$HOME/$incoming_arg")
case "$incoming" in
    "$root"/incoming/*) ;;
    *)
        echo "deploy: refusing unexpected incoming path: $incoming" >&2
        exit 2
        ;;
esac
case "$commit" in *[!0-9a-f]* | '') echo "deploy: invalid commit" >&2; exit 2 ;; esac
case "$dirty" in 0 | 1) ;; *) echo "deploy: dirty must be 0 or 1" >&2; exit 2 ;; esac
case "$restart" in 0 | 1) ;; *) echo "deploy: restart must be 0 or 1" >&2; exit 2 ;; esac

trap 'rm -rf -- "$incoming"' EXIT

src="$root/src"
mkdir -p "$src" "$root/bin" "$HOME/backups/lilia"
# The source tree keeps its own Pixi environment: the installed binaries load
# the GCC runtime, libuv and libcurl from it through their rpath.
rsync --recursive --links --checksum --delete --delete-delay \
    --exclude=/.pixi/ --exclude=/.xmake/ --exclude='/build*/' \
    "$incoming/" "$src/"

cd "$src"
export LILIA_SOURCE_COMMIT=$commit
export LILIA_SOURCE_DIRTY=$dirty
pixi run xmake f -m release --sanitizers=n -y
pixi run xmake build -y lilia

tag=${commit:0:12}
[[ $dirty == 1 ]] && tag+=-dirty
install -m 755 build/linux/x86_64/release/lilia "$root/bin/lilia-$tag"
ln -sfn "lilia-$tag" "$root/bin/current.new"
mv -T "$root/bin/current.new" "$root/bin/current"
echo "installed $root/bin/lilia-$tag"
"$root/bin/current" --version

unit_dir="$HOME/.config/systemd/user"
mkdir -p "$unit_dir"
if ! cmp -s deploy/lilia.service "$unit_dir/lilia.service"; then
    if [[ -f $unit_dir/lilia.service ]]; then
        backup="$HOME/backups/lilia/lilia.service-$(date +%Y%m%d-%H%M%S)"
        cp -p "$unit_dir/lilia.service" "$backup"
        echo "backed up the previous unit to $backup"
    fi
    install -m 644 deploy/lilia.service "$unit_dir/lilia.service"
    systemctl --user daemon-reload
    echo "installed $unit_dir/lilia.service"
fi

if [[ $restart == 1 ]]; then
    systemctl --user restart lilia.service
    sleep 2
    systemctl --user --no-pager --lines=0 status lilia.service || true
fi
