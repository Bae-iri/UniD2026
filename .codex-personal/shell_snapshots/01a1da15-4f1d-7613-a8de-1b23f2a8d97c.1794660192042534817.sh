# Snapshot file
# Unset all aliases to avoid conflicts with functions
unalias -a 2>/dev/null || true
shopt -u autocd
shopt -u assoc_expand_once
shopt -u cdable_vars
shopt -u cdspell
shopt -u checkhash
shopt -u checkjobs
shopt -s checkwinsize
shopt -s cmdhist
shopt -u compat31
shopt -u compat32
shopt -u compat40
shopt -u compat41
shopt -u compat42
shopt -u compat43
shopt -u compat44
shopt -s complete_fullquote
shopt -u direxpand
shopt -u dirspell
shopt -u dotglob
shopt -u execfail
shopt -u expand_aliases
shopt -u extdebug
shopt -u extglob
shopt -s extquote
shopt -u failglob
shopt -s force_fignore
shopt -s globasciiranges
shopt -u globstar
shopt -u gnu_errfmt
shopt -u histappend
shopt -u histreedit
shopt -u histverify
shopt -s hostcomplete
shopt -u huponexit
shopt -u inherit_errexit
shopt -s interactive_comments
shopt -u lastpipe
shopt -u lithist
shopt -u localvar_inherit
shopt -u localvar_unset
shopt -s login_shell
shopt -u mailwarn
shopt -u no_empty_cmd_completion
shopt -u nocaseglob
shopt -u nocasematch
shopt -u nullglob
shopt -s progcomp
shopt -u progcomp_alias
shopt -s promptvars
shopt -u restricted_shell
shopt -u shift_verbose
shopt -s sourcepath
shopt -u xpg_echo
# Functions
gawklibpath_append () 
{ 
    [ -z "$AWKLIBPATH" ] && AWKLIBPATH=`gawk 'BEGIN {print ENVIRON["AWKLIBPATH"]}'`;
    export AWKLIBPATH="$AWKLIBPATH:$*"
}
gawklibpath_default () 
{ 
    unset AWKLIBPATH;
    export AWKLIBPATH=`gawk 'BEGIN {print ENVIRON["AWKLIBPATH"]}'`
}
gawklibpath_prepend () 
{ 
    [ -z "$AWKLIBPATH" ] && AWKLIBPATH=`gawk 'BEGIN {print ENVIRON["AWKLIBPATH"]}'`;
    export AWKLIBPATH="$*:$AWKLIBPATH"
}
gawkpath_append () 
{ 
    [ -z "$AWKPATH" ] && AWKPATH=`gawk 'BEGIN {print ENVIRON["AWKPATH"]}'`;
    export AWKPATH="$AWKPATH:$*"
}
gawkpath_default () 
{ 
    unset AWKPATH;
    export AWKPATH=`gawk 'BEGIN {print ENVIRON["AWKPATH"]}'`
}
gawkpath_prepend () 
{ 
    [ -z "$AWKPATH" ] && AWKPATH=`gawk 'BEGIN {print ENVIRON["AWKPATH"]}'`;
    export AWKPATH="$*:$AWKPATH"
}

# setopts 3
set -o braceexpand
set -o hashall
set -o interactive-comments

# aliases 0

# exports (native declarations)
declare -x BROWSER="/home/ubuntu/.vscode-server/cli/servers/Stable-69047191fc427f2f92828279aae1d90d8eeda7ab/server/bin/helpers/browser.sh"
declare -x BUNDLED_DEBUGPY_PATH="/home/ubuntu/.vscode-server/extensions/ms-python.debugpy-2026.6.0-linux-x64/bundled/libs/debugpy"
declare -x CODEX_HOME="/home/ubuntu/JEONGYUN/.codex-personal"
declare -x COLORTERM="truecolor"
declare -x COPILOT_DEBUG_NONCE="2d2c397d52f20aed34726cafde4b063a"
declare -x DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/1000/bus"
declare -x GIT_ASKPASS="/home/ubuntu/.vscode-server/cli/servers/Stable-69047191fc427f2f92828279aae1d90d8eeda7ab/server/extensions/git/dist/askpass.sh"
declare -x HOME="/home/ubuntu"
declare -x LANG="en_US.UTF-8"
declare -x LESSCLOSE="/usr/bin/lesspipe %s %s"
declare -x LESSOPEN="| /usr/bin/lesspipe %s"
declare -x LOGNAME="ubuntu"
declare -x LS_COLORS="rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=00:su=37;41:sg=30;43:ca=30;41:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.arj=01;31:*.taz=01;31:*.lha=01;31:*.lz4=01;31:*.lzh=01;31:*.lzma=01;31:*.tlz=01;31:*.txz=01;31:*.tzo=01;31:*.t7z=01;31:*.zip=01;31:*.z=01;31:*.dz=01;31:*.gz=01;31:*.lrz=01;31:*.lz=01;31:*.lzo=01;31:*.xz=01;31:*.zst=01;31:*.tzst=01;31:*.bz2=01;31:*.bz=01;31:*.tbz=01;31:*.tbz2=01;31:*.tz=01;31:*.deb=01;31:*.rpm=01;31:*.jar=01;31:*.war=01;31:*.ear=01;31:*.sar=01;31:*.rar=01;31:*.alz=01;31:*.ace=01;31:*.zoo=01;31:*.cpio=01;31:*.7z=01;31:*.rz=01;31:*.cab=01;31:*.wim=01;31:*.swm=01;31:*.dwm=01;31:*.esd=01;31:*.jpg=01;35:*.jpeg=01;35:*.mjpg=01;35:*.mjpeg=01;35:*.gif=01;35:*.bmp=01;35:*.pbm=01;35:*.pgm=01;35:*.ppm=01;35:*.tga=01;35:*.xbm=01;35:*.xpm=01;35:*.tif=01;35:*.tiff=01;35:*.png=01;35:*.svg=01;35:*.svgz=01;35:*.mng=01;35:*.pcx=01;35:*.mov=01;35:*.mpg=01;35:*.mpeg=01;35:*.m2v=01;35:*.mkv=01;35:*.webm=01;35:*.webp=01;35:*.ogm=01;35:*.mp4=01;35:*.m4v=01;35:*.mp4v=01;35:*.vob=01;35:*.qt=01;35:*.nuv=01;35:*.wmv=01;35:*.asf=01;35:*.rm=01;35:*.rmvb=01;35:*.flc=01;35:*.avi=01;35:*.fli=01;35:*.flv=01;35:*.gl=01;35:*.dl=01;35:*.xcf=01;35:*.xwd=01;35:*.yuv=01;35:*.cgm=01;35:*.emf=01;35:*.ogv=01;35:*.ogx=01;35:*.aac=00;36:*.au=00;36:*.flac=00;36:*.m4a=00;36:*.mid=00;36:*.midi=00;36:*.mka=00;36:*.mp3=00;36:*.mpc=00;36:*.ogg=00;36:*.ra=00;36:*.wav=00;36:*.oga=00;36:*.opus=00;36:*.spx=00;36:*.xspf=00;36:"
declare -x MOTD_SHOWN="pam"
declare -x PATH="/home/ubuntu/.local/bin:/home/ubuntu/JEONGYUN/.codex-personal/tmp/arg0-linux/codex-arg0UF3GRt:/home/ubuntu/.codex/packages/standalone/releases/0.169.0-x86_64-unknown-linux-musl/codex-path:/home/ubuntu/.local/bin:/home/ubuntu/venv/bin:/home/ubuntu/.vscode-server/cli/servers/Stable-69047191fc427f2f92828279aae1d90d8eeda7ab/server/bin/remote-cli:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/home/ubuntu/.vscode-server/extensions/ms-python.debugpy-2026.6.0-linux-x64/bundled/scripts/noConfigScripts:/home/ubuntu/.vscode-server/data/User/globalStorage/github.copilot-chat/debugCommand:/home/ubuntu/.vscode-server/cli/servers/Stable-69047191fc427f2f92828279aae1d90d8eeda7ab/server/bin/copilot-shim"
declare -x PYDEVD_DISABLE_FILE_VALIDATION="1"
declare -x PYTHONSTARTUP="/home/ubuntu/.vscode-server/data/User/workspaceStorage/5865bbf134b6e3405e12f0b53569b84c/ms-python.python/pythonrc.py"
declare -x SHELL="/bin/bash"
declare -x SHLVL="3"
declare -x SSH_CLIENT="118.235.13.58 6964 22"
declare -x SSH_CONNECTION="118.235.13.58 6964 118.216.110.84 22"
declare -x TERM="xterm-256color"
declare -x TERM_PROGRAM="vscode"
declare -x TERM_PROGRAM_VERSION="1.146.0"
declare -x USER="ubuntu"
declare -x VIRTUAL_ENV="/home/ubuntu/venv"
declare -x VIRTUAL_ENV_PROMPT="(venv) "
declare -x VSCODE_DEBUGPY_ADAPTER_ENDPOINTS="/home/ubuntu/.vscode-server/extensions/ms-python.debugpy-2026.6.0-linux-x64/.noConfigDebugAdapterEndpoints/endpoint-ed5b6e4cc144d3e4.txt"
declare -x VSCODE_GIT_ASKPASS_EXTRA_ARGS=""
declare -x VSCODE_GIT_ASKPASS_MAIN="/home/ubuntu/.vscode-server/cli/servers/Stable-69047191fc427f2f92828279aae1d90d8eeda7ab/server/extensions/git/dist/askpass-main.js"
declare -x VSCODE_GIT_ASKPASS_NODE="/home/ubuntu/.vscode-server/cli/servers/Stable-69047191fc427f2f92828279aae1d90d8eeda7ab/server/node"
declare -x VSCODE_GIT_IPC_HANDLE="/run/user/1000/vscode-git-a2ed3aa4a1.sock"
declare -x VSCODE_IPC_HOOK_CLI="/run/user/1000/vscode-ipc-655163a0-8181-49d5-b63b-0b6fe5c55695.sock"
declare -x VSCODE_PYTHON_AUTOACTIVATE_GUARD="1"
declare -x XDG_DATA_DIRS="/usr/local/share:/usr/share:/var/lib/snapd/desktop"
declare -x XDG_RUNTIME_DIR="/run/user/1000"
declare -x XDG_SESSION_CLASS="user"
declare -x XDG_SESSION_ID="49"
declare -x XDG_SESSION_TYPE="tty"
