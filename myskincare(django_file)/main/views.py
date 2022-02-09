from django.shortcuts import render, redirect, get_object_or_404
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import konlpy
import nltk
from nltk.corpus import stopwords
from konlpy.tag import Twitter

#좋은/나쁜성분 리스트 생성
good_oily = '글리콜린산/살리실산/난 옥시놀-9/녹차/위치 하젤/레몬/캄파/멘톨클로로필/알라토인/티트리/감초/징크 옥사이트/칼렌 듈라 추출물/설퍼/트리클로잔/티타늄 옥사이드'
bad_oily = '트리글리세라이드/팔마티산염/미리스틴산/스테아르산염/스테아린산/코코넛오일/시어버터/바세린/옥시벤존/메톡시시나메이트'
good_dry = '히아루론산/글리세린/프로필렌 글라이콜/1,3-부틸렌 글라이콘/소디움PCA/비타민E/비타민A/비타민C/콜레젠/엘라스틴/아보카도 오일/이브닝 프라임 로즈 오일/오트밀 단백질/콩 추출물/카모마일/오이/복숭아/해조 추출물/상백피 추출물/코직산/알부틴/포토씨 추출물/베타카로틴/시어버터/파일워트 추출물/비타민B 복합체/판테놀'
bad_dry = '알코올/진흙/계면활성제/멘톨/페퍼민트'
good_sensitive = '비타민K/비타민F/호스트체스트넛 추출물/카모마일/알로에/콘플라워/알란토인/해조 추출물/티타늄 옥사이트'
bad_sensitive = '알코올/계면활성제/멘톨/페퍼민트/유칼립투스/아로마오일/고농도 과일산(AHA,BHA)/오렌지/딸기/레몬/레티놀/옥시벤존/메톡시 시나메이트'
bad_complex = bad_dry + bad_oily

def tfidf_vect_func(df, target_col, target_row):

    tfidf_vect = TfidfVectorizer()
    feature_vect = tfidf_vect.fit_transform(df[target_col])

    similarity_simple_pair = cosine_similarity(feature_vect[target_row], feature_vect)
    result_list = similarity_simple_pair.tolist()[0]
    df[f'{target_col}_result'] = result_list

    return df

def count_vect_func(df,target_col, target_row):
    count_vect_category = CountVectorizer(min_df=0, ngram_range=(1,2))
    feature_vect = count_vect_category.fit_transform(df[target_col])

    similarity_simple_pair = cosine_similarity(feature_vect[target_row], feature_vect)
    result_list = similarity_simple_pair.tolist()[0]
    df[f'{target_col}_result'] = result_list

    return df

def tfidf_count_result(df,review_col,tag_col,problem,hashtag):

    df = df.append(pd.Series(),ignore_index= True)
    last_index = df.iloc[[-1]].index[0]

    df.at[last_index, review_col] = problem  #input('피부 고민을 말씀해주세요. : ')\
    df.at[last_index,tag_col] = hashtag      #input('태그를 입력해주세요 : ')

    df = tfidf_vect_func(df, review_col, last_index)
    df = count_vect_func(df, tag_col, last_index)

    return df

def sort_review(df, sc):
    #tag 가중치
    df['review_result'] = df['review_result']+(df['product_type_result']*0.05)

    #상위10%잘라내기
    df_filt_review_sort = df.sort_values(by='review_result',ascending=False).copy()
    df_filt_review_sort = df_filt_review_sort[1:]
    df_filt_num = len(df_filt_review_sort)*0.1
    df_filt_num = int(df_filt_num)
    df_review_cut = df_filt_review_sort[:df_filt_num]

    #겹치는 리뷰 갯수 센 후 열 추가
    num_count_dict = df_review_cut['name'].value_counts().to_dict()
    df_review_cut['num_count'] = df_review_cut['name'].apply(lambda x : num_count_dict[x])

    #겹치는 행 삭제
    df_result2 = df_review_cut.drop_duplicates(['name'])

    #겹치는 리뷰 갯수 스케일링 후 가산점
    df_result2['num_count_score'] = sc.fit_transform(df_result2[['price_num']])
    df_result2['new_result'] = df_result2['review_result']+( 0.05*df_result2['num_count_score'])

    #new result 순으로 정렬
    df_result2.sort_values(by='new_result',ascending=False,inplace=True)
    return df_result2

def weight_sum (df,question_1,question_2,min_max_sc):
    #가격이 중요할 경우 가중치 부여
    if question_1 == 'q1_price' :
        # df['price_num_score'] = min_max_sc.fit_transform(df[['price_num']])
        df['new_result'] = df.apply(lambda x : x['new_result']+( 0.01*(1- x['price_num_score'])), axis=1)

        #성분이 중요할 경우 가중치 부여
    if question_2 == 'q2_ingre':
        # df['danger_num_score'] = min_max_sc.fit_transform(df[['danger_num']])
        df['new_result'] = df.apply(lambda x : x['new_result']+( 0.01*(1-x['danger_num_score'])), axis=1)

    df.sort_values(by='new_result',ascending=False,inplace=True)
    df_final = df

    return df_final

def choice_df(category):
    if category == 'cat_1':
        df = pd.read_excel('./main/data/cream_complete.xlsx')
    elif category == 'cat_2':
        df = pd.read_excel('./main/data/essence_sentiment_result.xlsx')
    elif category == 'cat_3':
        df = pd.read_excel('./main/data/lotion_sentiment_result.xlsx')
    elif category == 'cat_4':
        df = pd.read_excel('./main/data/mist_sentiment_result.xlsx')
    elif category == 'cat_5':
        df = pd.read_excel('./main/data/oil_sentiment_result.xlsx')
    elif category == 'cat_6':
        df = pd.read_excel('./main/data/skin_toner_sentiment_result.xlsx')

    return df

# html과 연동 코드 시작ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

def index(request):
    return render(request, 'main/index.html',{})

def recommend(request):
    # 선택된 DataFrame 변수에 저장
    df = choice_df(request.POST['category'])

    good_oily_list = good_oily.split('/')
    bad_oily_list = bad_oily.split('/')
    good_dry_list = good_dry.split('/')
    bad_dry_list = bad_dry.split('/')
    good_sensitive_list = good_sensitive.split('/')
    bad_sensitive_list = bad_sensitive.split('/')
    good_complex_list = []
    bad_complex_list = bad_complex.split('/')

    #피해야할 성분이 있는 행 삭제
    del_list = '|'.join(bad_oily_list)
    df_filt = df[~df['ingredient'].str.contains(del_list)]

    # 선택된 DataFrame의 유사도 분석 진행
    df_result = tfidf_count_result(df_filt,'review','product_type', request.POST['problem'], request.POST['hashtag'])
    # 유사도 분석 진행완료 후 data sort
    min_max_sc = MinMaxScaler()
    df_sort = sort_review(df_result, min_max_sc)

    #추천성분 가산점
    good_list = '|'.join(good_oily_list) #피부타입 입력값에따른 good리스트
    df_sort['good'] = df_sort['ingredient'].str.contains(good_list)
    df_sort['new_result'] = df_sort.apply(lambda x : x['new_result']+( 0.01*x['good']), axis=1)

    #감성분석 가산점
    df_sort['new_result'] = df_sort.apply(lambda x : x['new_result']+( 0.01*x['sentiment']), axis=1)

    df_sort_final = weight_sum(df_sort, request.POST['question_1'],request.POST['question_2'], min_max_sc)
    df_final = df_sort_final[:8]
    df_final = df_final.reset_index()

    product_list_1 = []
    product_list_2 = []

    for i in range(len(df_final)):
        if i >= 4:
            product_list_2.append(dict(df_final.loc[i,df_final.columns.values]))
        else:
            product_list_1.append(dict(df_final.loc[i,df_final.columns.values]))

    context = {'product_list_1' : product_list_1, 'product_list_2' : product_list_2} # 'product_list':product_list
    return render(request, 'main/recommend_result.html', context)

def product_detail(request):

    data = request.POST['data_dict']
    data_2 = eval(data)
    data_2['price_num']  =  format(int(data_2['price_num']), ',')
    data_2['sentiment_judge_rate'] = round((data_2['sentiment_judge_rate'])*100, 1)
    print(data_2)
    print(data_2['sentiment'])

    return render(request, 'main/product_detail.html', data_2)
