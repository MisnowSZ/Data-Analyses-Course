import requests
import json

query_text = '棉花糖'

headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36'}
url = 'https://www.douban.com'

def download(src, id):
	dir = './spider/' + str(id) + '.jpg'
	try:
		pic = requests.get(src, headers = headers, timeout = 30)
		#print(pic.status_code)
		fp = open(dir, 'wb')
		fp.write(pic.content)
		fp.close()
	except requests.exceptions.ConnectionError:
		print("Can not down load ", dir)

for i in range(0, 100, 20):
	url = 'https://www.douban.com/j/search_photo?q='+query_text+'&limit=20&start='+str(i)
	html = requests.get(url, headers = headers).text #reutrn result
	response = json.loads(html, encoding = 'utf-8') #convert JSON object to Python object
	#print(response)
	for image in response['images']:
		print(image['src'])
		download(image['src'], image['id'])

'''
rGet = requests.get(url, headers = headers)
print("rGet: ", type(rGet.text), rGet.text)

rPost = requests.post(url, data={'key':'value'}, headers = headers)
#print("rPost: ", type(rPost.text), rPost.text)
'''
