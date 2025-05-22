# Parto dall'immagine slim di python 3.13
FROM python:3.13-slim

# Imposta la directory di lavoro all'interno del container
WORKDIR /app

# Installa le dipendenze di sistema necessarie per Chrome e altre utilities richieste da Selenium
RUN apt-get update && apt-get install -y \
    libnss3 \
    libx11-xcb1 \
    libxcb1 \
    libgconf-2-4 \
    libfontconfig1 \
    chromium-driver \
    wget \
    gnupg \
    curl \
    unzip \
    && wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list \
    && apt-get update \
    && rm -rf /var/lib/apt/lists/*

# Scarica e installa Chrome
RUN wget -O /tmp/chrome.zip https://storage.googleapis.com/chrome-for-testing-public/136.0.7103.94/linux64/chrome-linux64.zip \
    && unzip /tmp/chrome.zip -d /opt/ \
    && mv /opt/chrome-linux64 /opt/chrome \
    && ln -s /opt/chrome/chrome /usr/bin/google-chrome \
    && rm /tmp/chrome.zip

# Scarica e installa chromedriver nella versione corrispondente a Chrome
RUN wget -O /tmp/chromedriver.zip https://storage.googleapis.com/chrome-for-testing-public/136.0.7103.94/linux64/chromedriver-linux64.zip \
    && unzip /tmp/chromedriver.zip -d /tmp/ \
    && mv /tmp/chromedriver-linux64/chromedriver /usr/bin/ \
    && chmod +x /usr/bin/chromedriver \
    && rm -rf /tmp/chromedriver*

# Copia i file necessari nell'immagine   
COPY ./app/requirements.txt /app/requirements.txt
COPY ./app /app

# Installa le dipendenze di python generate con `pip freeze > requirements.txt`
RUN pip install pip --upgrade
RUN pip install --no-cache-dir -r requirements.txt

# Espone la porta 8501 utilizzata da Streamlit
EXPOSE 8501

# Definisce il comando di avvio dell'applicazione Streamlit
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
