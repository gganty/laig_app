set -e

cd "$(dirname "$0")"

echo "Запускаю приложение..."
echo

if command -v python3 &> /dev/null; then
    PY=python3
    PIP=pip3
elif command -v python &> /dev/null; then
    PY=python
    PIP=pip
else
    echo "[ERROR] Python не найден. Установите Python 3 (https://python.org или пакетный менеджер)."
    exit 1
fi

if ! $PY -c "import streamlit" &> /dev/null; then
    echo "Streamlit не установлен. Установка зависимостей..."
    $PIP install -r requirements.txt
fi

echo "Открываю http://localhost:8501 ..."
echo "Для остановки нажмите Ctrl+C в этом окне."
echo

exec streamlit run app.py --server.headless false --browser.gatherUsageStats false
