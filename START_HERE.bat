@echo off
echo ============================================
echo   Setting up and starting the app...
echo   (first run installs packages, takes a
echo    minute or two; after that it's instant)
echo ============================================
echo.

pip install --quiet streamlit pandas numpy scikit-learn radon

echo.
echo Starting the app now - your browser will open automatically...
echo.

python -m streamlit run app.py

pause
