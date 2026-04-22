#!/bin/bash
# Football Agent Project - WSL Ubuntu environment setup
# Switches apt + pip to Tsinghua mirrors, installs gfootball build deps,
# creates a Python venv, and pip-installs gfootball.

set -e  # exit on first error

cd ~

echo "=== [1/5] Switching apt to Tsinghua mirror ==="
if [ -f /etc/apt/sources.list ]; then
  sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak
  sudo sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g; s|http://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list
fi
if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then
  sudo cp /etc/apt/sources.list.d/ubuntu.sources /etc/apt/sources.list.d/ubuntu.sources.bak
  sudo sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list.d/ubuntu.sources
fi

echo "=== [2/5] apt update ==="
sudo apt update

echo "=== [3/5] Installing gfootball build deps ==="
sudo apt install -y \
  git cmake build-essential \
  libgl1-mesa-dev libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev \
  libboost-all-dev \
  python3-pip python3-venv python3-dev

echo "=== [4/5] Configuring pip mirror ==="
mkdir -p ~/.pip
cat > ~/.pip/pip.conf <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF

echo "=== [5/5] Creating venv + installing gfootball ==="
python3 -m venv ~/football-env
source ~/football-env/bin/activate
pip install --upgrade pip setuptools wheel
pip install gfootball

echo ""
echo "=== Done! ==="
python3 -c "import gfootball; print('gfootball version:', gfootball.__version__)"
echo ""
echo "Activate the venv any time with:"
echo "  source ~/football-env/bin/activate"
