$CONTAINER_NAME = "drc_solver_env"

Write-Host "[Setup 1/2] Downloading Boxoban dataset as root..." -ForegroundColor Cyan
docker exec -u root $CONTAINER_NAME python -c "import urllib.request, zipfile, os; os.makedirs('/workspace/.sokoban_cache', exist_ok=True); urllib.request.urlretrieve('https://github.com/DeepMind/boxoban-levels/archive/master.zip', '/workspace/.sokoban_cache/master.zip'); zipfile.ZipFile('/workspace/.sokoban_cache/master.zip').extractall('/workspace/.sokoban_cache/'); os.remove('/workspace/.sokoban_cache/master.zip')"

Write-Host "[Setup 2/2] Downloading Hugging Face essential weights..." -ForegroundColor Cyan
docker exec $CONTAINER_NAME python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='AlignmentResearch/learned-planner', allow_patterns=['drc33/bkynosqi/cp_2002944000/*', 'drc33/bkynosqi/*.json', 'drc33/bkynosqi/*.txt'])"

Write-Host "==> Setup complete! You can now run: python drc_solver.py" -ForegroundColor Green