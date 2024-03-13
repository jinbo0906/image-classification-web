import streamlit as st
from PIL import Image
from clf import predict
import time

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("小狗图像分类预测应用")
st.write("")
st.write("")
option = st.selectbox(
    '模型选择',
    ('resnet50', 'resnet101', 'densenet121', 'shufflenet_v2_x0_5', 'mobilenet_v2'))
""

file_up = st.file_uploader("加载图片", type="jpg")

if file_up is not None:  # 检查文件是否已上传
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("稍等片刻...")
    labels, fps = predict(file_up, option)

    # print out the top 5 prediction labels with scores
    st.success('预测成功')
    for i in labels:
        st.write("预测结果 (index, name)", i[0], ",   Score: ", i[1])
else:
    st.warning('请选择一个图片文件')
