import insightface

model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=0)  # 0=GPU, -1=CPU
print("模型初始化完成")
source .venv/bin/activate
