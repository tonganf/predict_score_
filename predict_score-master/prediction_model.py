import pickle
import numpy as np
import pandas as pd
import streamlit as st
import time

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# loadding saved model
model = pickle.load(open('hsa_predict_model.sav', 'rb'))


# form nhap diem
st.title('Định dạng mẫu file dữ liệu')
st.markdown('Tải và mở file excel mẫu, nhập thông tin vào các ô có sẵn theo định dạng đã được thiết kế sẵn trong file.')

with open('file_nhap.xlsx', 'rb') as f:
    bytes_data = f.read()
    st.download_button(
        label='Tải xuống định dạng điểm',
        data=bytes_data,
        file_name='file_nhap.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

# Tạo một widget để tải file
# , accept=[".xlsx", ".xls"]
file_upload = st.file_uploader("Tải file mẫu vừa sửa lên", )


if file_upload is not None:
    data = pd.read_excel(file_upload)       # Đọc dữ liệu từ file
    # st.dataframe(data)                      # Hiển thị dữ liệu


def processing_input_data(x):
    lbl_gioitinh = preprocessing.LabelEncoder()
    lbl_gioitinh.fit(['Nam', 'Nữ'])
    x[0] = lbl_gioitinh.transform([x[0]])
    x[0] = x[0][0]
    lbl_hocluc = preprocessing.LabelEncoder()
    lbl_hocluc.fit(['Giỏi', 'Khá', 'Trung bình'])
    x[11] = lbl_hocluc.transform([x[11]])
    x[11] = x[11][0]
    x[22] = lbl_hocluc.transform([x[22]])
    x[22] = x[22][0]
    x[33] = lbl_hocluc.transform([x[33]])
    x[33] = x[33][0]
    x = np.array(x)
    d2x = np.reshape(x, (-1, 34))
    return d2x


def predict_hsa_grade(input_data):
    x = processing_input_data(input_data)
    hsa_score_pred = model.predict(x)
    return hsa_score_pred


def main():
    # app title
    st.title("Hệ thống dự đoán điểm thi HSA")

    gioitinh = st.selectbox(
        'Giới tính',
        ('Nam', 'Nữ'))

    col1, col2, col3 = st.columns(3)
    with col1:
        tongket10 = st.number_input(
            'Điểm nhập môn tin học', min_value=0.0, max_value=10.0, step=0.1)
        toan10 = st.number_input(
            'Điểm pháp luật đại cương', min_value=0.0, max_value=10.0, step=0.1)
        van10 = st.number_input('Điểm toán cao cấp 1',
                                min_value=0.0, max_value=10.0, step=0.1)
        ly10 = st.number_input('Điểm toán cao cấp 2',
                               min_value=0.0, max_value=10.0, step=0.1)
        hoa10 = st.number_input('Điểm toán rời rạc',
                                min_value=0.0, max_value=10.0, step=0.1)
        sinh10 = st.number_input(
            'Điểm triết học Mác-Lênin', min_value=0.0, max_value=10.0, step=0.1)
        su10 = st.number_input(
            'Điểm tiếng anh 1', min_value=0.0, max_value=10.0, step=0.1)
        # dia10 = st.number_input('Điểm tổng kết môn Địa lớp 10', min_value=0.0, max_value=10.0, step=0.1)
        # gdcd10 = st.number_input('Điểm tổng kết môn Công Dân lớp 10', min_value=0.0, max_value=10.0, step=0.1)
        # ngoaingu10 = st.number_input('Điểm tổng kết môn Ngoại Ngữ lớp 10', min_value=0.0, max_value=10.0, step=0.1)
        hocluc10 = st.selectbox(
            'Học lực kì I năm I',
            ('Giỏi', 'Khá', 'Trung bình'))
    with col2:
        tongket11 = st.number_input(
            'Điểm cơ sở dữ liệu', min_value=0.0, max_value=10.0, step=0.1)
        toan11 = st.number_input(
            'Điểm kiến trúc máy tính', min_value=0.0, max_value=10.0, step=0.1)
        van11 = st.number_input(
            'Điểm kinh tế  chính trị Mác-Lênin', min_value=0.0, max_value=10.0, step=0.1)
        ly11 = st.number_input('Điểm lập trình C nâng cao',
                               min_value=0.0, max_value=10.0, step=0.1)
        hoa11 = st.number_input(
            'Điểm tổng kết môn Hóa lớp 11', min_value=0.0, max_value=10.0, step=0.1)
        sinh11 = st.number_input(
            'Điểm tổng kết môn Sinh lớp 11', min_value=0.0, max_value=10.0, step=0.1)
        su11 = st.number_input('Điểm tổng kết môn Sử lớp 11',
                               min_value=0.0, max_value=10.0, step=0.1)
        dia11 = st.number_input(
            'Điểm tổng kết môn Địa lớp 11', min_value=0.0, max_value=10.0, step=0.1)
        gdcd11 = st.number_input(
            'Điểm tổng kết môn Công Dân lớp 11', min_value=0.0, max_value=10.0, step=0.1)
        ngoaingu11 = st.number_input(
            'Điểm tổng kết môn Ngoại Ngữ lớp 11', min_value=0.0, max_value=10.0, step=0.1)
        hocluc11 = st.selectbox(
            'Học lực lớp 11',
            ('Giỏi', 'Khá', 'Trung bình'))
    with col3:
        tongket12 = st.number_input(
            'Điểm tổng kết lớp 12', min_value=0.0, max_value=10.0, step=0.1)
        toan12 = st.number_input(
            'Điểm tổng kết môn Toán lớp 12', min_value=0.0, max_value=10.0, step=0.1)
        van12 = st.number_input(
            'Điểm tổng kết môn Văn lớp 12', min_value=0.0, max_value=10.0, step=0.1)
        ly12 = st.number_input(
            'Điểm tổng kết môn Vật Lý lớp 12', min_value=0.0, max_value=10.0, step=0.1)
        hoa12 = st.number_input(
            'Điểm tổng kết môn Hóa lớp 12', min_value=0.0, max_value=10.0, step=0.1)
        sinh12 = st.number_input(
            'Điểm tổng kết môn Sinh lớp 12', min_value=0.0, max_value=10.0, step=0.1)
        su12 = st.number_input('Điểm tổng kết môn Sử lớp 12',
                               min_value=0.0, max_value=10.0, step=0.1)
        dia12 = st.number_input(
            'Điểm tổng kết môn Địa lớp 12', min_value=0.0, max_value=10.0, step=0.1)
        gdcd12 = st.number_input(
            'Điểm tổng kết môn Công Dân lớp 12', min_value=0.0, max_value=10.0, step=0.1)
        ngoaingu12 = st.number_input(
            'Điểm tổng kết môn Ngoại Ngữ lớp 12', min_value=0.0, max_value=10.0, step=0.1)
        hocluc12 = st.selectbox(
            'Học lực lớp 12',
            ('Giỏi', 'Khá', 'Trung bình'))

    arr = [gioitinh,
           tongket10, toan10, van10, ly10, hoa10, sinh10, su10, hocluc10,
           tongket11, toan11, van11, ly11, hoa11, sinh11, su11, dia11, gdcd11, ngoaingu11, hocluc11,
           tongket12, toan12, van12, ly12, hoa12, sinh12, su12, dia12, gdcd12, ngoaingu12, hocluc12]

    results = []    # Khai báo biến chứa kết quả dự đoán (nếu lớn hơn 2 record)

    if file_upload is not None:
        for index in range(len(data)):
            res = predict_hsa_grade(data.iloc[index].values)
            results.append(res.round(0))
    else:
        res = predict_hsa_grade(arr)[0]

    if st.button('Dự đoán kết quả HSA'):
        with st.spinner('Wait for it...'):
            time.sleep(2)
        if len(results) == 1:
            st.success(
                'Kết quả dự đoán điểm thi HSA của bạn là: ' + str(int(res)))
        if len(results) > 1:
            df_res = pd.DataFrame(results)
            df_res.columns = ['Điểm HSA dự đoán']
            df_res = pd.concat([data, df_res], axis=1)

            df_res.to_excel('predict_score.xlsx', index=False)
            with open('predict_score.xlsx', 'rb') as f:
                bytes_data = f.read()
            st.download_button(
                label='Tải xuống tệp Excel',
                data=bytes_data,
                file_name='predict_score.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )


if __name__ == '__main__':
    main()
