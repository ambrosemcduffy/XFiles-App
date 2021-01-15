import requests
from bs4 import BeautifulSoup

# using this header to prevent mod security error
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0)'}

url = "http://www.clivebanks.co.uk/X-Files%20Timeline.htm"
site_url = "/".join(url.split("/")[:3])

response = requests.get(url, headers=headers)

# Obtaining the front page as writting it on disk instead memory.
with open("front_page.html", mode="wb") as file:
    file.write(response.content)

# Getting the links of all front page
with open("front_page.html") as file:
    soup = BeautifulSoup(file, features="html.parser")

# Looping through links of season one.
urls = []
for a in soup.find_all("a", href=True):
    urls.append(site_url+"/"+a["href"])
    if a["href"] == "X-Files/Erlenmeyerflask.htm":
        break

# Saving out the text file to disk as a dataset.
with open("x_files_dataset.txt", mode="a") as file:
    for url_ in urls[1:]:
        response = requests.get(url_, headers=headers)
        soup = BeautifulSoup(response.content, features="html.parser")
        file.write(soup.extract().text)
