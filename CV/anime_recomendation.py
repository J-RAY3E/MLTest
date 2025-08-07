import requests
from bs4 import BeautifulSoup


def getpages(link):
    main_page = requests.get(link)
    main_page = BeautifulSoup(main_page.text,"html.parser")
    for page_n in main_page.find("ul",class_ = "pagination").find_all("li")[::-1]:
        text = page_n.find("a").text
        if text.isdecimal():
            return int(text)
        

list_animes = []
link = "https://www3.animeflv.net/perfil/J-ray/favoritos"
num_pages = getpages(link)
for num_page in range(num_pages+1):
    page = requests.get(f"{link}?page={num_page}")
    if page.status_code == 404:
        break
    page_pased = BeautifulSoup(page.text,"html.parser")
    ul = page_pased.find("ul",class_ = "ListAnimes")
    for li in ul.find_all("li"):
        title = li.find("div",class_ = "Title").find("a").get_text()
        list_animes.append(title)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"

def build_prompt(list_animes):
    return f"""
    Eres un experto en recomendaciones de animes. A continuación se te dará una lista de animes que el usuario ya ha visto.

    Tu tarea es:
    1. Analizar los géneros, tramas y temas comunes de estos animes.
    2. Recomendar **5 animes nuevos** que **no estén en la lista**.
    3. Que sean similares en tono, género y estilo narrativo.
    4. Solo responde con los nombres, uno por línea. Nada más.

    Lista de animes vistos:
    {list_animes}

    Recomendaciones:
    """.strip()

prompt  = build_prompt(list_animes)


response = requests.post(
    OLLAMA_URL,
    json={"model": MODEL, "prompt": prompt, "stream": False},
    timeout=60
)


'''

class LLM():
    def __init__(self,username  = "" , page_name =  "AnimeFLV"):
        super().__init__
        self.username = username
        self.page_name = page_name
        self.listAnimes =  []
        
class MINNER():
    def __init__(self,username  = "" , page_name =  "AnimeFLV"):
        super().__init__
        self.username = username
        self.page_name = page_name
        self.listAnimes =  []
    def LoadList(self):
        #function to load list of favorite animes from page_name
        pass
    def getList(self):
        return self.listAnimes
    def clean(self):
        self.listAnimes.clear()
    def addAnime(self,anime:str):
        self.listAnimes.append(anime)
'''