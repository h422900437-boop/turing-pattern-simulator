import cupy as cp

# def test_gpu():
#     try:
#         print("正在尝试唤醒 GPU...")
#         # 让 GPU 在显存中生成一个 3x3 的矩阵
#         x_gpu = cp.arange(9).reshape(3, 3)
#         # 让 GPU 做一次矩阵乘法
#         y_gpu = x_gpu * 2
        
#         print("✅ CuPy 安装成功！GPU 通讯完全畅通！")
#         print(f"使用的设备: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')}")
#         print("GPU 计算出的矩阵结果为：")
#         print(y_gpu)
#     except Exception as e:
#         print("❌ 测试失败，错误信息如下：")
#         print(e)

# if __name__ == "__main__":
#     test_gpu()



def test_gpu():
    try:
        print("Attempting to wake up the GPU...")
        # Let the GPU generate a 3x3 matrix in VRAM
        x_gpu = cp.arange(9).reshape(3, 3)
        # Let the GPU perform a multiplication operation
        y_gpu = x_gpu * 2
        
        print("✅ CuPy installed successfully! GPU communication is fully open!")
        print(f"Device in use: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')}")
        print("The matrix result calculated by the GPU is:")
        print(y_gpu)
    except Exception as e:
        print("❌ Test failed. Error message below:")
        print(e)

if __name__ == "__main__":
    test_gpu()