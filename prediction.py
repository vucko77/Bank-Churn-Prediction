
model = load("model.h5")

answer = model.predict(testX)

print(answer)
