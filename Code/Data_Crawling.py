import os
import time
import urllib.parse
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager 


driver = webdriver.Chrome(ChromeDriverManager().install())

ls = ['cặp sách', 'áo thun', 'xe máy honda', 'điện thoại', 'máy tính bảng', 'máy vi tính', 'tai nghe', 'nồi cơm', 'nước hoa', 'đồng hồ', 'sách','vali', 'dụng cụ thể thao', 'nệm', 'xong chảo', 'nồi cơm']

for clas in ls:
    ds = []
    dir1 = r'project/class-' +  clas
    os.mkdir(dir1)
    for i in range(1,31): 
        parse = urllib.parse.quote(clas)
        url = 'https://shopee.vn/search?keyword='+ parse+'&page=' + str(i)
        driver.get(url)

        time.sleep(5) 
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight*0.25);") 
        time.sleep(2)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight*0.5);") 
        time.sleep(2)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight*0.75);") 
        time.sleep(2)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") 
        time.sleep(5)

        ls = driver.find_elements_by_class_name('_1T9dHf') 
        for el in ls:
            ds.append(el.get_attribute('src'))
    
    
    for i in range(1,len(ds)+1):
        opener = urllib.request.build_opener()
        urllib.request.install_opener(opener)
        try:
            filename, headers = urllib.request.urlretrieve(url=ds[i-1],filename= 'class-' + clas + "/" +str(i) + ".jfif" )
        except:
            pass
    time.sleep(5)
driver.close()