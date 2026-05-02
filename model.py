import numpy as np
import matplotlib.pyplot as plt 

#Generate simple data 
np.random.seed(42)
X = np.random.rand(100)
y = 3 * X + np.random.randn(100) * 0.1 

#Fit linear model 
coeffs = np.polyfit(X, y, 1)
y_pred = coeffs[0] * X + coeffs[1]

#Plot results 
plt.scatter(X, y, label="Actual")
plt.plot(X, y_pred, color='red', label="Predicted")
plt.legend()
plt.title("Model Results")

plt.savefig("model_results.png")
plt.close()

#Calculate metric 
mse = np.mean((y - y_pred) ** 2)

#Save metrics 
with open("metrics.txt", "w") as f: 
    f.write(f"MSE: {mse:.4f}\n")
    
print("Model ran successfully")