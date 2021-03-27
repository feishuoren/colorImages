import base64
import requests
 
# client_id 为获取的AK， client_secret 为获取的SK
AK = '5DsVYclkhO1MaYECj3dW9NvD'
SK = 'KLHXNXN77gWPAMrjXuMa45csCCjMwT5E'
host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id='+AK+'&client_secret='+SK
response = requests.get(host)
# if response:
#     print(response.json())
 
# 请求URL
request_url = "https://aip.baidubce.com/rest/2.0/image-process/v1/colourize"
# 二进制方式打开图片文件
f = open('./test.jpg', 'rb')
img = base64.b64encode(f.read())
# 请求参数
params = {"image":img}
access_token = response.json()['access_token']
request_url = request_url + "?access_token=" + access_token
headers = {'content-type': 'application/x-www-form-urlencoded'}
# 发送请求并获得响应
response = requests.post(request_url, data=params, headers=headers)
# if response:
#    print(response.json())
 
# base64编码转图片并存储
img = base64.b64decode(response.json()['image'])
file = open('./result.jpg', 'wb')
file.write(img)
file.close()

