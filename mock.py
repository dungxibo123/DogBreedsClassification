import time

def mock_train():
    for i in range(100):
        print(i, end='\t')
        time.sleep(0.05)
