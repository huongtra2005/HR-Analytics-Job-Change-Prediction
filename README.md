# Job Change Prediction using NumPy

## Giới thiệu
**Bài toán:** Dự đoán khả năng thay đổi công việc của một ứng viên dựa trên các thông tin cá nhân và công ty.  
**Động lực:** Giúp công ty giữ nhân viên tiềm năng, tối ưu chiến lược nhân sự.  
**Mục tiêu:**  
- Tiền xử lý dữ liệu bằng NumPy  
- Xây dựng mô hình Logistic Regression và KNN từ đầu  
- Đánh giá mô hình bằng Accuracy và F1-score  

---
## Vấn đề thực tế
Một công ty Data Science tổ chức nhiều khóa đào tạo cho các ứng viên. Trong số những người tham gia:  
- **Nhóm 1:** Thật sự muốn làm việc cho công ty sau khi hoàn thành khóa học  
- **Nhóm 2:** Chỉ muốn học để tìm công việc mới ở nơi khác  

**Mục tiêu:** Dự đoán xem ứng viên có khả năng thuộc nhóm nào để công ty có thể:  
- Xác định ứng viên tiềm năng  
- Điều chỉnh chương trình đào tạo và chính sách retention  

---

## Dataset
- **Nguồn dữ liệu:** HR Analytics: Job Change of Data Scientists  

| Feature | Mô tả |
|---------|-------|
| enrollee_id | ID ứng viên |
| city | Mã thành phố |
| city_development_index | Chỉ số phát triển thành phố (0–1) |
| gender | Giới tính |
| relevent_experience | Ứng viên có kinh nghiệm liên quan hay không |
| enrolled_university | Loại chương trình đại học đang theo học |
| education_level | Trình độ học vấn |
| major_discipline | Chuyên ngành đại học |
| experience | Tổng số năm kinh nghiệm |
| company_size | Quy mô công ty hiện tại |
| company_type | Loại công ty hiện tại |
| last_new_job | Thời gian giữa hai công việc gần nhất |
| training_hours | Tổng số giờ training đã hoàn thành |
| target | Mục tiêu: muốn đổi việc hay không (0 = không, 1 = có) | 

---
**Kích thước:** Dataset gồm 19158 ứng viên và 14 cột thông tin.

## Method
**Quy trình xử lý dữ liệu:**  
1. Điền giá trị missing bằng median  
2. Chuẩn hóa dữ liệu numeric bằng Min-Max normalization  
3. Thêm bias term cho Logistic Regression
4. Chỉ giữ lại các thông tin city_development_index, training_hours, experience, last_new_job, company_size, company_type, target

**Thuật toán sử dụng:**  

**Logistic Regression**:  
- Hàm sigmoid:  
\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]  
- Gradient descent cập nhật theta:  
\[
\theta := \theta - \alpha \cdot \frac{1}{m} X^T (\sigma(X \theta) - y)
\]

**KNN:**  
- Dự đoán dựa trên k điểm gần nhất theo khoảng cách Euclidean:  
\[
dist(x_i, x_j) = \sqrt{\sum_{f=1}^{n} (x_i^f - x_j^f)^2}
\]  
---

## Installation & Setup
1. Download 23120177.zip về máy
2. Cài môi trường venv
3. cd đến folder chứa requirements.txt
4. install toàn bộ thư viện trong file requirements.txt
5. Vào notebook chạy file ipynb

## Usage
Chạy notebook notebooks/modeling.ipynb để:
- Train Logistic Regression & KNN
- Đánh giá Accuracy, F1-score
  
## Results
- Logistic Regression:
    * Train Accuracy: 75%
    * Train F1: 0.006
    * Test Accuracy: 75%
    * Test F1: 0.008
- KNN:
    * Test Accuracy: 76%
    * Test F1: 0.48
- Nhận xét:
    * Logistic Regression dự đoán hầu hết nhãn 0 → F1-score thấp
    * KNN dự đoán nhãn 1 tốt hơn → F1-score cân bằng hơn
      
## Project Structure:
HR-Analytics-Job-Change-Prediction/
├── README.md
├── requirements.txt
├── data/			
│   ├── raw/           	# Dữ liệu gốc
│   └── processed/      # Dữ liệu đã xử lý
├── notebooks/
│   ├── data_exploring.ipynb
│   ├── preprocessing.ipynb
│   └── modeling.ipynb

## Challenges & Solutions
- Dữ liệu lệch nhãn: Logistic Regression dự đoán hầu hết là 0 → F1-score thấp
- Giải pháp: Sử dụng KNN; cân nhắc class weighting; feature engineering
  
## Future Improvements
- Thử các mô hình ensemble: Random Forest, XGBoost
- Oversampling nhãn thiểu số (SMOTE)
- Hyperparameter tuning để cải thiện F1-score
- 
## Contributors
Họ và tên: Phạm Hương Trà
MSSV: 23120177
Contact: 23120177@student.hcmus.edu.vn