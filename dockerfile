FROM od 

USER root

WORKDIR /data
COPY . .

# 修改項目資料夾的權限，讓 tensorflow 用戶可以訪問
RUN mkdir -p /data/projects && chown -R tensorflow:tensorflow /data/projects

RUN apt-get install -y libgl1-mesa-glx

# 切換回 tensorflow 用戶
USER tensorflow

# 安裝 Gradio
RUN pip install --no-cache-dir gradio

# 暴露 Gradio 預設使用的端口 7860
EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"

# 執行 main.py
CMD ["python", "main.py"]


