import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Đọc dữ liệu từ file CSV
df = pd.read_csv('sale_data.csv')

# Kiểm tra số lượng dòng ban đầu
initial_rows = df.shape[0]
print(f"Initial number of rows: {initial_rows}")

# Kiểm tra giá trị NaN trong cột TotalAmount trước khi chuyển đổi
nan_before_conversion = df['TotalAmount'].isna().sum()
print(f"NaN values in TotalAmount before conversion: {nan_before_conversion}")

# Chuyển đổi cột TotalAmount thành kiểu số thực
df['TotalAmount'] = pd.to_numeric(df['TotalAmount'], errors='coerce')

# Kiểm tra giá trị NaN trong cột TotalAmount sau khi chuyển đổi
nan_after_conversion = df['TotalAmount'].isna().sum()
print(f"NaN values in TotalAmount after conversion: {nan_after_conversion}")

# Loại bỏ các giá trị NaN nếu có sau khi chuyển đổi
df = df.dropna(subset=['TotalAmount'])

# Kiểm tra số lượng dòng sau khi loại bỏ giá trị NaN
rows_after_dropping_nan = df.shape[0]
print(f"Number of rows after dropping NaN values: {rows_after_dropping_nan}")

# Kiểm tra số lượng dòng trong cột TotalAmount
total_amount_rows = df['TotalAmount'].count()
print(f"Number of rows in TotalAmount column: {total_amount_rows}")

# Chọn các cột và chuẩn bị dữ liệu
X = df[['Quantity', 'Discount']]
y = df['TotalAmount']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Tính toán lỗi dự đoán
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Vẽ biểu đồ so sánh giữa giá trị thực tế và dự đoán
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Actual vs Predicted Total Amount')
plt.xlabel('Actual Total Amount')
plt.ylabel('Predicted Total Amount')
plt.show()
