#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/micace/Project/AI-3D"
BACKUP_DIR="/home/micace/backup-AI-3D-$(date +%Y%m%d)"
GITHUB_REPO_NAME="AI-3D"

cd "$PROJECT_DIR"

# ──────────────────────────────────────
echo "=== BƯỚC 0: Cập nhật .gitignore ==="
# Đảm bảo các file lớn KHÔNG bị git theo dõi
cat > .gitignore << 'GITIGNORE_EOF'
# --- Python / build ---
.venv/
__pycache__/
*.py[cod]
*.egg-info/
build/
dist/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# --- Data, models, and large artifacts (kept local) ---
data/
outputs/
submissions/
checkpoints/
logs/
workspace/
models/
configs/local/

# --- Vendored / external repos ---
third_party/

# --- CodeGraph index (regenerable) ---
.codegraph/

# --- Local editor/system files ---
.DS_Store
*.swp
*.swo
.idea/
.vscode/
GITIGNORE_EOF

# ──────────────────────────────────────
echo "=== BƯỚC 1: Push source code lên GitHub ==="

# Xác thực GitHub nếu chưa có
if ! gh auth status &>/dev/null; then
    echo "Cần đăng nhập GitHub:"
    gh auth login
fi

# Tạo remote nếu chưa có
if ! git remote get-url origin &>/dev/null; then
    echo "Tạo GitHub repository: $GITHUB_REPO_NAME"
    gh repo create "$GITHUB_REPO_NAME" --private --source=. --remote=origin --push || {
        echo "Tạo repo thất bại. Hãy tạo thủ công tại https://github.com/new"
        echo "Repo name: $GITHUB_REPO_NAME"
        echo "Rồi chạy: git remote add origin <url> && git push -u origin master"
        exit 1
    }
else
    git add -A
    if git diff --cached --quiet; then
        echo "  Không có thay đổi mới."
    else
        git commit -m "Backup $(date +%Y-%m-%d)"
        git push -u origin master
    fi
    echo "  GitHub: done"
fi

# ──────────────────────────────────────
echo "=== BƯỚC 2: Nén các thư mục lớn ==="
mkdir -p "$BACKUP_DIR"

ZIP_DIRS=(
    "checkpoints"
    "data"
    "outputs"
    "submissions"
    "workspace"
    "reports"
)

for dir in "${ZIP_DIRS[@]}"; do
    if [ -d "$dir" ] && [ "$(du -s "$dir" | cut -f1)" -gt 1000 ]; then
        size=$(du -sh "$dir" | cut -f1)
        echo "  Nén $dir ($size) ..."
        (cd "$PROJECT_DIR" && zip -r -q "$BACKUP_DIR/${dir}.zip" "$dir" \
            -x "*/node_modules/*" "*/__pycache__/*" "*.pyc" "*/venv/*" "*/.*" 2>/dev/null) &
    else
        echo "  Bỏ qua $dir (không tồn tại hoặc nhỏ)"
    fi
done
wait
echo "  Hoàn tất nén."

echo ""
echo "=== KẾT QUẢ ==="
du -sh "$BACKUP_DIR"/*.zip 2>/dev/null
echo ""
echo "Backup tại: $BACKUP_DIR"
echo "Tổng dung lượng: $(du -sh "$BACKUP_DIR" | cut -f1)"
echo ""

# ──────────────────────────────────────
echo "=== BƯỚC 3: Upload lên Google Drive ==="
if rclone listremotes 2>/dev/null | grep -q "gdrive:"; then
    echo "  Đang upload..."
    rclone copy "$BACKUP_DIR/" "gdrive:AI-3D-backup/" --progress
    echo "  Google Drive: done"
else
    echo "=== CẦN CẤU HÌNH RCLONE CHO GOOGLE DRIVE ==="
    echo "Chạy lệnh sau (chỉ 1 lần):"
    echo ""
    echo "  rclone config"
    echo ""
    echo "Làm theo hướng dẫn:"
    echo "  1. Chọn 'n' (new remote)"
    echo "  2. Name: gdrive"
    echo "  3. Type: drive"
    echo "  4. client_id: (enter để skip)"
    echo "  5. client_secret: (enter để skip)"
    echo "  6. Chọn scope: 1 (drive.file)"
    echo "  7. root_folder_id: (enter)"
    echo "  8. service_account_file: (enter)"
    echo "  9. Chọn 'n' (Edit advanced config)"
    echo " 10. Chọn 'y' (Auto config) -> trình duyệt mở ra, đăng nhập Google"
    echo " 11. Chọn 'y' (quota)"
    echo " 12. Chọn 'q' (quit)"
    echo ""
    echo "Sau đó chạy lại:"
    echo "  rclone copy $BACKUP_DIR/ gdrive:AI-3D-backup/ --progress"
fi

echo ""
echo "=== HOÀN TẤT ==="
