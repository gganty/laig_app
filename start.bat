@echo off
chcp 65001 > nul
title Интерактивное приложение для подготовки к КЛ2-ЛАиГ.

echo Запускаю приложение...
echo.

python --version > nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python не найден. Установите Python с https://python.org
    pause
    exit /b 1
)

python -c "import streamlit" > nul 2>&1
if errorlevel 1 (
    echo Streamlit не установлен. Установка зависимостей...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Не удалось установить зависимости.
        pause
        exit /b 1
    )
)

echo Открываю http://localhost:8501 ...
echo Для остановки закрыть это окно или нажать Ctrl+C.
echo.

streamlit run app.py --server.headless false --browser.gatherUsageStats false

pause
