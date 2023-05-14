#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# In[7]:


import streamlit as st
import pandas as pd
import pickle
from pickle import dump, load
import os
import re
import pysrt
import zipfile
import requests
import spacy
import string
import en_core_web_sm

from PIL import Image
from zipfile import ZipFile
from datetime import datetime
from requests import Session, RequestException
from typing import Optional, Union

from urllib.parse import urljoin
from bs4 import BeautifulSoup, Tag, NavigableString


MAIN_URL = 'https://www.opensubtitles.org/'

def find_tag(
        session: BeautifulSoup,
        tag: str,
        attrs: dict[str]
    ) -> Optional[Union[Tag, NavigableString]]:
    result = session.find(name=tag, attrs=attrs)
    if result is None:
        raise Exception(
        st.sidebar.warning(f'Тег {tag}, с параметрами {attrs} не найден. Уточните, пожалуйста, введенные данные.', icon="⚠️")
        )
    return result


# In[8]:


def parse_srt_url(
        title: str, year: int,
        season: Optional[int] = None,
        episode: Optional[int] = None,
        is_series: bool = False,) -> tuple[str]:
    movie_name = '+'.join(title.split())
    link = (f"/en/search2/sublanguageid-eng/searchonlymovies-on/"
            f"movieyearsign-1/movieyear-{year}/"
            f"moviename-{movie_name}")
    if is_series:
        link = (f'/en/search2/sublanguageid-eng/searchonlytvseries-on/'
                f'season-{season}/episode-{episode}/'
                f'movieyearsign-1/'
                f'movieyear-{year}/moviename-{movie_name}')
    link = urljoin(MAIN_URL, link)
    session = Session()
    try:
        response = session.get(link)
        response.encoding = 'utf-8'
    except RequestException:
        raise RequestException(
            st.sidebar.warning(f'Возникла ошибка при загрузке страницы {link}. Уточните, пожалуйста, введенные данные.', icon="⚠️")
            )
    soup = BeautifulSoup(response.text, features='lxml')
    if 'subtitle-language-selector-link' not in soup:
        movie_link = find_tag(soup, 'a', {'class': 'bnone'})['href']
        response = session.get(urljoin(MAIN_URL, movie_link))
        soup = BeautifulSoup(response.text, 'lxml')
    movie_name = soup.find(
            'a', {'class': 'bt-dwl external adds_trigger'}
        )
    if movie_name is not None:
        movie_name = movie_name['data-product-title']
    else:
        movie_name = find_tag(soup, 'a', attrs={'class': 'bnone'}).text
    searched_tag = find_tag(
        soup, 'a', {'href': re.compile(r'/en/subtitleserve/sub/\d+$')}
    )
    poster_link = soup.find('img', attrs={'alt': 'film'})
    poster_link = urljoin(MAIN_URL, poster_link['src'])
    str_link = urljoin(MAIN_URL, searched_tag['href'])
    return movie_name.replace("\n", ""), poster_link, str_link



st.title('Узнайте уровень сложности английского языка фильма(сериала) 🎥')

st.subheader('Введите, пожалуйста, данные о фильме(сериале)')
warning = 'Проверьте, пожалуйста, правильность введенных данных!'

type_film = st.radio("Выберите тип ", options=("Фильм", "Сериал"), key='type_film')
if type_film == 'Фильм':
    title = st.text_input("Полное название фильма на английском языке", key='title') 
    year = st.number_input('Год релиза', key='year')
    if 1920 > year < datetime.now().year:
        st.info(warning, icon="⚠️")
    season = 0
    episode = 0
else:   
    title = st.text_input("Введите, пожалуйста, полное название сериала на английском языке", key='title') 
    year = st.number_input('Год релиза', key='year')
    if 1920 > year < datetime.now().year:
        st.info(warning, icon="⚠️")    
    season = st.number_input('Сезон', key='season')
    episode = st.number_input('Серия', key='episode')


year = int(year)
season = int(season)
episode = int(episode)


is_series = False
if type_film == "Сериал":
    is_series = True
    
if st.button("Определить уровень английского языка"):
    name, path_img, path = parse_srt_url(title=title, year=year, season=season, episode=episode, is_series=is_series)


    HTML = r'<.*?>' # html тэги меняем на пробел
    TAG = r'{.*?}' # тэги меняем на пробел
    COMMENTS = r'[\(\[][A-Za-z ]+[\)\]]' # комменты в скобках меняем на пробел
    UPPER = r'[[A-Za-z ]+[\:\]]' # указания на того кто говорит (BOBBY:)
    LETTERS = r'[^a-zA-Z\'.,!? ]' # все что не буквы меняем на пробел 
    SPACES = r'([ ])\1+' # повторяющиеся пробелы меняем на один пробел
    DOTS = r'[\.]+' # многоточие меняем на точку
    SYMB = r"[^\w\d'\s]" # знаки препинания кроме апострофа

    def clean_subs(subs):
        subs = subs[1:] # удаляем первый рекламный субтитр
        txt = re.sub(HTML, ' ', subs.text) # html тэги меняем на пробел
        txt = re.sub(COMMENTS, ' ', txt) # комменты в скобках меняем на пробел
        txt = re.sub(UPPER, ' ', txt) # указания на того кто говорит (BOBBY:)
        txt = re.sub(LETTERS, ' ', txt) # все что не буквы меняем на пробел
        txt = re.sub(DOTS, r'.', txt) # многоточие меняем на точку
        txt = re.sub(SPACES, r'\1', txt) # повторяющиеся пробелы меняем на один пробел
        txt = re.sub(SYMB, '', txt) # знаки препинания кроме апострофа на пустую строку
        txt = re.sub('www', '', txt) # кое-где остаётся www, то же меняем на пустую строку
        txt = txt.lstrip() # обрезка пробелов слева
        txt = txt.encode('ascii', 'ignore').decode() # удаляем все что не ascii символы   
        txt = txt.lower() # текст в нижний регистр
        return txt    

    #nlp = spacy.load("en_core_web_sm")
    nlp = en_core_web_sm.load()


    punctuations = string.punctuation

    stopwords = spacy.lang.en.stop_words.STOP_WORDS

    def spacy_tokenizer(sentence):
        tokens = nlp(sentence)

        tokens = [word.lemma_.lower().strip() if word.lemma_ != "PROPN" else word.lower_ for word in tokens]
    
        tokens = [word for word in tokens if word not in stopwords and word not in punctuations]

        return tokens


    #with open('C:/Users/Olga/Documents/Masterskay_2/model.pkl', 'rb') as dataset:
    #    model = pickle.load(dataset)
    with open(os.path.dirname(__file__) + '/model.pkl', 'rb') as dataset: 
        model = pickle.load(dataset)



    r = requests.get(path)
    with open("minemaster1.zip", "wb") as code:
        code.write(r.content)
    r_img = requests.get(path_img)
    with open("image.jpg", "wb") as code:
        code.write(r_img.content)    
    
    with zipfile.ZipFile("minemaster1.zip", 'r') as zip_ref:
        zip_ref.extractall()    
    
    sub = clean_subs(pysrt.open(zipfile.ZipFile("minemaster1.zip").namelist()[0], encoding='iso-8859-1'))

    #data = pd.DataFrame({'Subtitles': sub}, index=[0])
    data = pd.Series(sub)

    prediction = model.predict(data)
    st.sidebar.header('Результаты',)
    st.sidebar.write(f'Уровень сложности английского языка определен для: {name}' )
    #st.sidebar.write(f'Уровень сложности английского языка: {(str(prediction).strip("[]"))}')
    st.sidebar.write(f"Уровень сложности английского языка: {''.join(prediction)}")
    image = Image.open("image.jpg")
    st.sidebar.image(image)
    st.balloons()  
    
    os.remove(zipfile.ZipFile("minemaster1.zip").namelist()[0])
    os.remove(zipfile.ZipFile("minemaster1.zip").namelist()[1])
    os.remove("minemaster1.zip")
    os.remove( "image.jpg")
    
    #os.remove(os.path.dirname(__file__) + '/'+ (zipfile.ZipFile("minemaster1.zip").namelist()[0]))
    #os.remove(os.path.dirname(__file__) + '/'+ (zipfile.ZipFile("minemaster1.zip").namelist()[1]))
    #os.remove(os.path.dirname(__file__) + '/'+ "minemaster1.zip")
    #os.remove(os.path.dirname(__file__) + '/'+ "image.jpg")
                              


# In[ ]:




