from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen-VL-Chat",
    local_dir="/lustre/hdd/LAS/jannesar-lab/tmahmud/models/qwen-vl-chat",
    local_dir_use_symlinks=False,
)