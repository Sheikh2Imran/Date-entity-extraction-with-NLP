import nltk
import datetime
from datetime import date
from datetime import timedelta
from nltk import RegexpTagger
from nltk import word_tokenize, pos_tag
import dateutil.relativedelta as rdelta
import pandas as pd 

patterns = [
(r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  #cardinal numbers
(r'(sunday|SUNDAY|Sunday|sun|SUN|Sun|Monday|MONDAY|monday|Mon|MON|mon|Tuesday|TUESDAY|tuesday|Tues|TUES|tues|Wednesday|WEDNESDAY|wednesday|Wed|WED|wed|Thursday|THURSDAY|thursday|Thurs|thurs|THURS|Friday|FRIDAY|friday|fri|Fri|FRI|Saturday|SATURDAY|saturday|Sat|sat|SAT)$','DAYS'),
(r'(January|january|JANUARY|February|february|FEBRUARY|March|MARCH|march|April|APRIL|april|May|may|MAY|June|june|JUNE|JULY|July|july|August|AUGUST|august|September|Sep|SEPTEMBER|september|October|OCTOBER|october|November|NOVEMBER|november|December|DECEMBER|december)$','MONTH'), #month
(r'(JAN|jan|Jan|Feb|FEB|feb|Mar|MAR|mar|Apr|apr|APR|May|may|MAY|JUN||jun|Jun|jul|JUL|Jul|Aug|aug|AUG|Sep|sep|SEP|oct|Oct|OCT|Nov|nov|NOV|Dec|dec|DEC)$','month'), #month

(r'([12]\d{3}(-|/|.)(0[1-9]|1[0-2])(-|/|.)([1-9]|0[1-9]|[12]\d|3[01]))', 'DATE'),
(r'([12]\d{1}(-|/|.)(0[1-9]|1[0-2])(-|/|.)([1-9]|0[1-9]|[12]\d|3[01]))', 'DATE'),
(r'((0[1-9]|1[0-2])(-|/|.)(0[1-9]|1[0-2])(-|/|.)(0[1-9]|[12]\d|3[01]))', 'DATE'),
(r'([12]\d{1}(-|/|.)([1-9]|0[1-9]|1[0-2])(-|/|.)([1-9]|0[1-9]|[12]\d|3[01]))', 'DATE'),
(r'((0[1-9]|1[0-2])(-|/|.)([1-9]|0[1-9]|1[0-2])(-|/|.)([1-9]|0[1-9]|[12]\d|3[01]))', 'DATE'),
(r'([12]\d{3}(-|/|.)([1-9]|0[1-9]|1[0-2])(-|/|.)(0[1-9]|[12]\d|3[01]))', 'DATE'),

(r'((0[1-9]|[12]\d|3[01])(-|/|.)(0[1-9]|1[0-2])(-|/|.)([12]\d{3}))', 'DATE'),
(r'((0[1-9]|[12]\d|3[01])(-|/|.)(0[1-9]|1[0-2])(-|/|.)[12]\d{1})', 'DATE'),
(r'((0[1-9]|[12]\d|3[01])(-|/|.)(0[1-9]|1[0-2])(-|/|.)(0[1-9]|1[0-2]))', 'DATE'),
(r'(([1-9]|0[1-9]|[12]\d|3[01])(-|/|.)([1-9]|0[1-9]|1[0-2])(-|/|.)[12]\d{3})', 'DATE'),
(r'(([1-9]|0[1-9]|[12]\d|3[01])(-|/|.)([1-9]|0[1-9]|1[0-2])(-|/|.)[12]\d{1})', 'DATE'),
(r'(([1-9]|0[1-9]|[12]\d|3[01])(-|/|.)([1-9]|0[1-9]|1[0-2])(-|/|.)(0[1-9]|1[0-2]))', 'DATE'),

(r'(([1-9]|0[1-9]|1[0-2])(-|/|.)([1-9]|0[1-9]|[12]\d|3[01])(-|/|.)[12]\d{3})', 'DATE'),
(r'(([1-9]|0[1-9]|1[0-2])(-|/|.)([1-9]|0[1-9]|[12]\d|3[01])(-|/|.)[12]\d{1})', 'DATE'),
(r'(([1-9]|0[1-9]|1[0-2])(-|/|.)([1-9]|0[1-9]|[12]\d|3[01])(-|/|.)(0[1-9]|1[0-2]))', 'DATE'),
(r'((0[1-9]|1[0-2])(-|/|.)([1-9]|0[1-9]|[12]\d|3[01])(-|/|.)[12]\d{1})', 'DATE'),
(r'((0[1-9]|1[0-2])(-|/|.)([1-9]|0[1-9]|[12]\d|3[01])(-|/|.)(0[1-9]|1[0-2]))', 'DATE'),
(r'((0[1-9]|1[0-2])(-|/|.)([1-9]|0[1-9]|[12]\d|3[01])(-|/|.)[12]\d{3})', 'DATE'),

(r'\d{1}-(January|january|JANUARY|Jan|jan|JAN|February|february|FEBRUARY|Feb|feb|FEB|March|march|MARCH|Mar|mar|MAR|April|april|APRIL|Apr|apr|APR|May|may|MAY|June|june|JUNE|Jun|jun|JUN|July|july|JULY|Jul|jul|JUL|August|august|AUGUST|Aug|aug|AUG|September|september|SEPTEMBER|Sep|sep|SEP|October|october|OCTOBER|Oct|oct|OCT|November|november|NOVEMBER|Nov|nov|NOV|December|december|DECEMBER|Dec|dec|DEC)-\d{2}', 'DATE_1_2'),
(r'\d{1}-(January|january|JANUARY|Jan|jan|JAN|February|february|FEBRUARY|Feb|feb|FEB|March|march|MARCH|Mar|mar|MAR|April|april|APRIL|Apr|apr|APR|May|may|MAY|June|june|JUNE|Jun|jun|JUN|July|july|JULY|Jul|jul|JUL|August|august|AUGUST|Aug|aug|AUG|September|september|SEPTEMBER|Sep|sep|SEP|October|october|OCTOBER|Oct|oct|OCT|November|november|NOVEMBER|Nov|nov|NOV|December|december|DECEMBER|Dec|dec|DEC)-\d{4}', 'DATE_1_4'),
(r'\d{2}-(January|january|JANUARY|Jan|jan|JAN|February|february|FEBRUARY|Feb|feb|FEB|March|march|MARCH|Mar|mar|MAR|April|april|APRIL|Apr|apr|APR|May|may|MAY|June|june|JUNE|Jun|jun|JUN|July|july|JULY|Jul|jul|JUL|August|august|AUGUST|Aug|aug|AUG|September|september|SEPTEMBER|Sep|sep|SEP|October|october|OCTOBER|Oct|oct|OCT|November|november|NOVEMBER|Nov|nov|NOV|December|december|DECEMBER|Dec|dec|DEC)-\d{2}', 'DATE_2_2'),
(r'\d{2}-(January|january|JANUARY|Jan|jan|JAN|February|february|FEBRUARY|Feb|feb|FEB|March|march|MARCH|Mar|mar|MAR|April|april|APRIL|Apr|apr|APR|May|may|MAY|June|june|JUNE|Jun|jun|JUN|July|july|JULY|Jul|jul|JUL|August|august|AUGUST|Aug|aug|AUG|September|september|SEPTEMBER|Sep|sep|SEP|October|october|OCTOBER|Oct|oct|OCT|November|november|NOVEMBER|Nov|nov|NOV|December|december|DECEMBER|Dec|dec|DEC)-\d{4}', 'DATE_2_4'),
(r'(1st|2nd|3rd|4th|5th|6th|7th|8th|9th|10th|11th|12th|13th|14th|15th|16th|17th|18th|19th|20th|21st|22nd|23rd|24th|25th|26th|27th|28th|29th|30th|31st)$', 'DATEUSP'), #DATEUSP
(r'.*', 'NN') # nouns (default)
]


cols = [1] 
df = pd.read_csv('test.csv', index_col=0, usecols=cols) 

for i, row in enumerate(df.values): 
    text = df.index[i]      
    
    reg_tagger = RegexpTagger(patterns) 
    tokens = word_tokenize(text)      #Tokenization of every word of a sentence
    
    #tagged = reg_tagger.tag(data)      #adding parts of speech tag to each text word
    tagged = reg_tagger.tag(tokens)   #adding parts of speech tag to each token
    
    #print(tagged)
    
    tag0 = [e[0] for e in tagged]     #gaining the first value of a list
    
    tag1 = [e[1] for e in tagged]     #gaining the second value of a list
    
    #print(tag0)
    #print(tag1)

    i=0
    Date = [] 
    list_len = len(tag1) 
    
    while(i < list_len):
        if(tag1[i]=="DATE"):
            Date.append(tag0[i])
        #for 21-january-2018 or 21-jan-18
        if(tag1[i]=="DATE_1_2" or tag1[i]=="DATE_1_4" or tag1[i]=="DATE_2_2" or tag1[i]=="DATE_2_4"):
            data = tag0[i]
            data1 = data.split('-')
            if(data1[1]=="January" or data1[1]=="january" or data1[1]=="JANUARY" or data1[1]=="Jan" or data1[1]=="jan" or data1[1]=="JAN"):
                data1[1]=1
                day = str(data1[0])
                month = str(data1[1])
                year = str(data1[2])
                entity = day+"-"+month+"-"+year
                Date.append(entity)
            elif(data1[1]=="February" or data1[1]=="february" or data1[1]=="FEBRUARY" or data1[1]=="Feb" or data1[1]=="feb" or data1[1]=="FEB"):
                data1[1]=2
                day = str(data1[0])
                month = str(data1[1])
                year = str(data1[2])
                entity = day+"-"+month+"-"+year
                Date.append(entity)     
            elif(data1[1]=="March" or data1[1]=="march" or data1[1]=="MARCH" or data1[1]=="Mar" or data1[1]=="mar" or data1[1]=="MAR"):
                data1[1]=3
                day = str(data1[0])
                month = str(data1[1])
                year = str(data1[2])
                entity = day+"-"+month+"-"+year
                Date.append(entity)
            elif(data1[1]=="April" or data1[1]=="april" or data1[1]=="APRIL" or data1[1]=="Apr" or data1[1]=="apr" or data1[1]=="APR"):
                data1[1]=4
                day = str(data1[0])
                month = str(data1[1])
                year = str(data1[2])
                entity = day+"-"+month+"-"+year
                Date.append(entity) 
            elif(data1[1]=="May" or data1[1]=="may" or data1[1]=="MAY"):
                data1[1]=5
                day = str(data1[0])
                month = str(data1[1])
                year = str(data1[2])
                entity = day+"-"+month+"-"+year
                Date.append(entity)
            elif(data1[1]=="June" or data1[1]=="june" or data1[1]=="JUNE" or data1[1]=="Jun" or data1[1]=="jun" or data1[1]=="JUN"):
                data1[1]=6
                day = str(data1[0])
                month = str(data1[1])
                year = str(data1[2])
                entity = day+"-"+month+"-"+year
                Date.append(entity)
            elif(data1[1]=="July" or data1[1]=="july" or data1[1]=="JULY" or data1[1]=="Jul" or data1[1]=="jul" or data1[1]=="JUL"):
                data1[1]=7
                day = str(data1[0])
                month = str(data1[1])
                year = str(data1[2])
                entity = day+"-"+month+"-"+year
                Date.append(entity)
            elif(data1[1]=="August" or data1[1]=="august" or data1[1]=="AUGUST" or data1[1]=="Aug" or data1[1]=="aug" or data1[1]=="AUG"):
                data1[1]=8
                day = str(data1[0])
                month = str(data1[1])
                year = str(data1[2])
                entity = day+"-"+month+"-"+year
                Date.append(entity)
            elif(data1[1]=="September" or data1[1]=="september" or data1[1]=="SEPTEMBER" or data1[1]=="Sep" or data1[1]=="sep" or data1[1]=="SEP"):
                data1[1]=9
                day = str(data1[0])
                month = str(data1[1])
                year = str(data1[2])
                entity = day+"-"+month+"-"+year
                Date.append(entity) 
            elif(data1[1]=="October" or data1[1]=="october" or data1[1]=="OCTOBER" or data1[1]=="Oct" or data1[1]=="oct" or data1[1]=="OCT"):
                data1[1]=10
                day = str(data1[0])
                month = str(data1[1])
                year = str(data1[2])
                entity = day+"-"+month+"-"+year
                Date.append(entity)
            elif(data1[1]=="November" or data1[1]=="november" or data1[1]=="NOVEMBER" or data1[1]=="Nov" or data1[1]=="nov" or data1[1]=="NOV"):
                data1[1]=11
                day = str(data1[0])
                month = str(data1[1])
                year = str(data1[2])
                entity = day+"-"+month+"-"+year
                Date.append(entity)
            elif(data1[1]=="December" or data1[1]=="december" or data1[1]=="DECEMBER" or data1[1]=="Dec" or data1[1]=="dec" or data1[1]=="DEC"):
                data1[1]=12
                day = str(data1[0])
                month = str(data1[1])
                year = str(data1[2])
                entity = day+"-"+month+"-"+year
                Date.append(entity)
                
        #for 21 january 2018 and 21 jan 2018
        if(tag1[i-1]=="CD"):
            if(tag1[i]=="MONTH"):
                date_formate = tag0[i-1]+" "+tag0[i]+" "+tag0[i+1]
                if(len(tag0[i+1]) == 2):
                    date_time_obj = datetime.datetime.strptime(date_formate, '%d %B %y')
                    ab= date_time_obj.date()
                    entity = datetime.datetime.strftime(ab, "%d-%m-%Y")
                    Date.append(entity)
                elif(len(tag0[i+1]) == 4):
                    date_time_obj = datetime.datetime.strptime(date_formate, '%d %B %Y')
                    ab= date_time_obj.date()
                    entity = datetime.datetime.strftime(ab, "%d-%m-%Y")
                    Date.append(entity)
                
            if(tag1[i]=="month"):
                date_formate = tag0[i-1]+" "+tag0[i]+" "+tag0[i+1]
                if(len(tag0[i+1]) == 2):
                    date_time_obj = datetime.datetime.strptime(date_formate, '%d %b %y')
                    ab= date_time_obj.date()
                    entity = datetime.datetime.strftime(ab, "%d-%m-%Y")
                    Date.append(entity)
                elif(len(tag0[i+1]) == 4):
                    date_time_obj = datetime.datetime.strptime(date_formate, '%d %b %Y')
                    ab= date_time_obj.date()
                    entity = datetime.datetime.strftime(ab, "%d-%m-%Y")
                    Date.append(entity)
          
        if(tag1[i-1]=="DATEUSP"):
            #for 21st january 2018 and 21st jan 2018
            if(tag0[i-1].endswith("st")==True):
                if(tag1[i]=="MONTH"):
                    date_formate = tag0[i-1]+" "+tag0[i]+" "+tag0[i+1]
                    if(len(tag0[i+1]) == 2):
                        date_time_obj = datetime.datetime.strptime(date_formate, '%dst %B %y')
                        ab= date_time_obj.date()
                        entity = datetime.datetime.strftime(ab, "%d-%m-%Y")
                        Date.append(entity)
                    elif(len(tag0[i+1]) == 4):
                        date_time_obj = datetime.datetime.strptime(date_formate, '%dst %B %Y')
                        ab= date_time_obj.date()
                        entity = datetime.datetime.strftime(ab, "%d-%m-%Y")
                        Date.append(entity)
                if(tag1[i]=="month"):
                    date_formate = tag0[i-1]+" "+tag0[i]+" "+tag0[i+1]
                    if(len(tag0[i+1]) == 2):
                        date_time_obj = datetime.datetime.strptime(date_formate, '%dst %b %y')
                        ab= date_time_obj.date()
                        entity = datetime.datetime.strftime(ab, "%d-%m-%Y")
                        Date.append(entity)
                    elif(len(tag0[i+1]) == 4):
                        date_time_obj = datetime.datetime.strptime(date_formate, '%dst %b %Y')
                        ab= date_time_obj.date()
                        entity = datetime.datetime.strftime(ab, "%d-%m-%Y")
                        Date.append(entity)
            
            #for 22nd january 2018 and 22nd jan 2018
            elif(tag0[i-1].endswith("nd")==True):
                if(tag1[i]=="MONTH"):
                    date_formate = tag0[i-1]+" "+tag0[i]+" "+tag0[i+1]
                    if(len(tag0[i+1]) == 2):
                        date_time_obj = datetime.datetime.strptime(date_formate, '%dnd %B %y')
                        ab= date_time_obj.date()
                        entity = datetime.datetime.strftime(ab, "%d-%m-%Y")
                        Date.append(entity)
                    elif(len(tag0[i+1]) == 4):
                        date_time_obj = datetime.datetime.strptime(date_formate, '%dnd %B %Y')
                        ab= date_time_obj.date()
                        entity = datetime.datetime.strftime(ab, "%d-%m-%Y")
                        Date.append(entity)
                if(tag1[i]=="month"):
                    date_formate = tag0[i-1]+" "+tag0[i]+" "+tag0[i+1]
                    if(len(tag0[i+1]) == 2):
                        date_time_obj = datetime.datetime.strptime(date_formate, '%dnd %b %y')
                        ab= date_time_obj.date()
                        entity = datetime.datetime.strftime(ab, "%d-%m-%Y")
                        Date.append(entity)
                    elif(len(tag0[i+1]) == 4):
                        date_time_obj = datetime.datetime.strptime(date_formate, '%dnd %b %Y')
                        ab= date_time_obj.date()
                        entity = datetime.datetime.strftime(ab, "%d-%m-%Y")
                        Date.append(entity)
            
            #for 23rd january 2018 and 23rd jan 2018
            elif(tag0[i-1].endswith("rd")==True):
                if(tag1[i]=="MONTH"):
                    date_formate = tag0[i-1]+" "+tag0[i]+" "+tag0[i+1]
                    if(len(tag0[i+1]) == 2):
                        date_time_obj = datetime.datetime.strptime(date_formate, '%drd %B %y')
                        ab= date_time_obj.date()
                        entity = datetime.datetime.strftime(ab, "%d-%m-%Y")
                        Date.append(entity)
                    elif(len(tag0[i+1]) == 4):
                        date_time_obj = datetime.datetime.strptime(date_formate, '%drd %B %Y')
                        ab= date_time_obj.date()
                        entity = datetime.datetime.strftime(ab, "%d-%m-%Y")
                        Date.append(entity)
                if(tag1[i]=="month"):
                    date_formate = tag0[i-1]+" "+tag0[i]+" "+tag0[i+1]
                    if(len(tag0[i+1]) == 2):
                        date_time_obj = datetime.datetime.strptime(date_formate, '%drd %b %y')
                        ab= date_time_obj.date()
                        entity = datetime.datetime.strftime(ab, "%d-%m-%Y")
                        Date.append(entity)
                    elif(len(tag0[i+1]) == 4):
                        date_time_obj = datetime.datetime.strptime(date_formate, '%drd %b %Y')
                        ab= date_time_obj.date()
                        entity = datetime.datetime.strftime(ab, "%d-%m-%Y")
                        Date.append(entity)
            
            #for 24th january 2018 and 24th jan 2018
            elif(tag0[i-1].endswith("th")==True):
                if(tag1[i]=="MONTH"):
                    date_formate = tag0[i-1]+" "+tag0[i]+" "+tag0[i+1]
                    if(len(tag0[i+1]) == 2):
                        date_time_obj = datetime.datetime.strptime(date_formate, '%dth %B %y')
                        ab= date_time_obj.date()
                        entity = datetime.datetime.strftime(ab, "%d-%m-%Y")
                        Date.append(entity)
                    elif(len(tag0[i+1]) == 4):
                        date_time_obj = datetime.datetime.strptime(date_formate, '%dth %B %Y')
                        ab= date_time_obj.date()
                        entity = datetime.datetime.strftime(ab, "%d-%m-%Y")
                        Date.append(entity)
                if(tag1[i]=="month"):
                    date_formate = tag0[i-1]+" "+tag0[i]+" "+tag0[i+1]
                    if(len(tag0[i+1]) == 2):
                        date_time_obj = datetime.datetime.strptime(date_formate, '%dth %b %y')
                        ab= date_time_obj.date()
                        entity = datetime.datetime.strftime(ab, "%d-%m-%Y")
                        Date.append(entity)
                    elif(len(tag0[i+1]) == 4):
                        date_time_obj = datetime.datetime.strptime(date_formate, '%dth %b %Y')
                        ab= date_time_obj.date()
                        entity = datetime.datetime.strftime(ab, "%d-%m-%Y")
                        Date.append(entity)
    
        #last/privous year -> year -> month -> week
        if(tag0[i]=="last" or tag0[i]=="Last" or tag0[i]=="LAST" or tag0[i]=="previous" or tag0[i]=="Previous" or tag0[i]=="PREVIOUS"):
            if(tag0[i+1]=="year" or tag0[i+1]=="Year" or tag0[i+1]=="YEAR"):
                now = datetime.datetime.now()
                now_to_last_year = now.year - 1
                last_year = str(now_to_last_year)
                entity = "01-01-"+last_year
                Date.append(entity)
            elif(tag0[i+1]=="month" or tag0[i+1]=="Month" or tag0[i+1]=="MONTH"):
                now = datetime.datetime.now()
                this_year = str(now.year)
                now_to_last_month = (now.month - 1)%12
                last_month = str(now_to_last_month)
                entity = "01-"+last_month+"-"+this_year
                Date.append(entity)
            elif(tag0[i+1]=="week" or tag0[i+1]=="Week" or tag0[i+1]=="WEEK"):
                today = datetime.datetime.today()
                next_week = today-datetime.timedelta(weeks=1)
                entity = datetime.datetime.strftime(next_week, "%d-%m-%Y")
                Date.append(entity)
                
        #this year -> this month -> this week
        if(tag0[i]=="This" or tag0[i]=="this" or tag0[i]=="THIS"):
            if(tag0[i+1]=="year" or tag0[i+1]=="Year" or tag0[i+1]=="YEAR"):
                now = datetime.datetime.now()
                this_year = str(now.year)
                entity = "01-01-"+this_year
                Date.append(entity)
            elif(tag0[i+1]=="month" or tag0[i+1]=="Month" or tag0[i+1]=="MONTH"):
                now = datetime.datetime.now()
                pre_month = str(now.month)
                pre_year = str(now.year)
                entity = "01-"+pre_month+"-"+pre_year
                Date.append(entity)
            elif(tag0[i+1]=='week' or tag0[i+1]=='Week' or tag0[i+1]=='WEEK'):
                entity = datetime.datetime.now().strftime("%d-%m-%Y")
                Date.append(entity)
    
        #next year -> month -> week
        if(tag0[i]=="next" or tag0[i]=="Next" or tag0[i]=="NEXT"):
            if(tag0[i+1]=="year" or tag0[i+1]=="Year" or tag0[i+1]=="YEAR"):
                now = datetime.datetime.now()
                now_to_next_year = now.year + 1
                next_year = str(now_to_next_year)
                entity = "01-01-"+next_year
                Date.append(entity)
            elif(tag0[i+1]=="month" or tag0[i+1]=="Month" or tag0[i+1]=="MONTH"):
                now = datetime.datetime.now()
                this_year = now.year
                now_to_next_month = now.month + 1
                if(now_to_next_month > 12):
                    next_month = now_to_next_month % 12
                    year = this_year + 1
                    entity = "01-"+str(next_month)+"-"+str(year)
                    Date.append(entity)
                else:
                    next_month = str(now_to_next_month)
                    entity = "01-"+str(now_to_next_month)+"-"+str(this_year)
                    Date.append(entity)
            elif(tag0[i+1]=="week" or tag0[i+1]=="Week" or tag0[i+1]=="WEEK"):
                today = datetime.datetime.today()
                next_week = today+datetime.timedelta(weeks=1)
                entity = datetime.datetime.strftime(next_week, "%d-%m-%Y")
                Date.append(entity)
            
        #today   
        if(tag0[i]=='today' or tag0[i]=='Today' or tag0[i]=='TODAY' or 
        (tag0[i]=="present" and tag0[i+1]=="day") or (tag0[i]=="Present" and tag0[i+1]=="day") or 
        (tag0[i]=="Present" and tag0[i+1]=="Day") or 
        (tag0[i]=="PRESENT" and tag0[i+1]=="DAY")):
            entity = datetime.datetime.now().strftime("%d-%m-%Y")
            Date.append(entity)
    
        #day after tomorrow/tomorrow
        if((tag0[i-1]=='after' and tag0[i]=="tomorrow") or 
        (tag0[i-1]=='after' and tag0[i]=="tomorrow") or 
        (tag0[i-1]=='After' and tag0[i]=="Tomorrow") or 
        (tag0[i-1]=='AFTER' and tag0[i]=="TOMORROW") or 
        (tag0[i-1]=='after' and tag0[i]=="tomorrow") or 
        (tag0[i-1]=='after' and tag0[i]=="tomorrow") or 
        (tag0[i-1]=='After' and tag0[i]=="Tomorrow") or 
        (tag0[i-1]=='AFTER' and tag0[i]=="TOMORROW")):
            today = datetime.datetime.today()
            yesterday = today+datetime.timedelta(2)
            entity = datetime.datetime.strftime(yesterday, "%d-%m-%Y")
            Date.append(entity)
        else:  
            if((tag0[i]=='tomorrow' or tag0[i]=='Tomorrow' or tag0[i]=='TOMORROW') or 
            (tag0[i]=="next" and tag0[i+1]=="day") or 
            (tag0[i]=="Next" and tag0[i+1]=="day") or 
            (tag0[i]=="Next" and tag0[i+1]=="Day") or 
            (tag0[i]=="NEXT" and tag0[i+1]=="DAY") or 
            (tag0[i]=="the" and tag0[i+1]=="following" and tag0[i+2]=="day") or 
            (tag0[i]=="The" and tag0[i+1]=="following" and tag0[i+2]=="day") or 
            (tag0[i]=="The" and tag0[i+1]=="Following" and tag0[i+2]=="Day") or 
            (tag0[i]=="THE" and tag0[i+1]=="FOLLOWING" and tag0[i+2]=="DAY")):
                today = datetime.datetime.today()
                tomorrow = today+datetime.timedelta(days=1)
                entity = datetime.datetime.strftime(tomorrow, "%d-%m-%Y")
                Date.append(entity)
    
        #yesterday
        if((tag0[i]=="yesterday" or tag0[i]=="Yesterday" or tag0[i]=="YESTERDAY") or 
        (tag0[i]=="the" and tag0[i+1]=="previous" and tag0[i+2]=="day") or 
        (tag0[i]=="The" and tag0[i+1]=="previous" and tag0[i+2]=="day") or 
        (tag0[i]=="The" and tag0[i+1]=="Previous" and tag0[i+2]=="Day") or 
        (tag0[i]=="THE" and tag0[i+1]=="PREVIOUS" and tag0[i+2]=="DAY") or 
        (tag0[i]=="last" and tag0[i+1]=="day") or 
        (tag0[i]=="Last" and tag0[i+1]=="Day") or 
        (tag0[i]=="LAST" and tag0[i+1]=="DAY")):
            today = datetime.datetime.today()
            yesterday = today-datetime.timedelta(days=1)
            entity = datetime.datetime.strftime(yesterday, "%d-%m-%Y")
            Date.append(entity)
    
        if(tag1[i]=='DAYS'):   
            #next monday -> tuesday -> wednesday -> thursday -> friday -> saturday -> sunday
            if(tag0[i-1]=='next' or tag0[i-1]=="Next" or tag0[i-1]=="NEXT"):
                if(tag0[i]=='mon' or tag0[i]=='Mon' or tag0[i]=='MON' or tag0[i]=='monday' or tag0[i]=='Monday' or tag0[i]=='MONDAY'):
                    today = date.today()
                    next_monday = today + rdelta.relativedelta(days=1, weekday=rdelta.MO(+1))
                    entity = datetime.datetime.strftime(next_monday, "%d-%m-%Y")
                    Date.append(entity)
    
                elif(tag0[i]=='tue' or tag0[i]=='Tue' or tag0[i]=='TUE' or tag0[i]=='tues' or tag0[i]=='Tues' or tag0[i]=='TUES' or tag0[i]=='tuesday' or tag0[i]=='Tuesday' or tag0[i]=='TUESDAY'):
                    today = date.today()
                    next_tueday = today + rdelta.relativedelta(days=1, weekday=rdelta.TU(+1))
                    entity = datetime.datetime.strftime(next_tueday, "%d-%m-%Y")
                    Date.append(entity)
    
                elif(tag0[i]=='wed' or tag0[i]=='Wed' or tag0[i]=='WED' or tag0[i]=='wednesday' or tag0[i]=='Wednesday' or tag0[i]=='WEDNESDAY'):
                    today = date.today()
                    next_wedday = today + rdelta.relativedelta(days=1, weekday=rdelta.WE(+1))
                    entity = datetime.datetime.strftime(next_wedday, "%d-%m-%Y")
                    Date.append(entity)
    
                elif(tag0[i]=='thu' or tag0[i]=='Thu' or tag0[i]=='THU' or tag0[i]=='thurs' or tag0[i]=='Thurs' or tag0[i]=='THURS' or tag0[i]=='thursday' or tag0[i]=='Thursday' or tag0[i]=='THURSDAY'):
                    today = date.today()
                    next_thuday = today + rdelta.relativedelta(days=1, weekday=rdelta.TH(+1))
                    entity = datetime.datetime.strftime(next_thuday, "%d-%m-%Y")
                    Date.append(entity)
    
                elif(tag0[i]=='fri' or tag0[i]=='Fri' or tag0[i]=='FRI' or tag0[i]=='friday' or tag0[i]=='Friday' or tag0[i]=='FRIDAY'):
                    today = date.today()
                    next_friday = today + rdelta.relativedelta(days=1, weekday=rdelta.FR(+1))
                    entity = datetime.datetime.strftime(next_friday, "%d-%m-%Y")
                    Date.append(entity)
    
                elif(tag0[i]=='sat' or tag0[i]=='Sat' or tag0[i]=='SAT' or tag0[i]=='saturday' or tag0[i]=='Saturday' or tag0[i]=='SATURDAY'):
                    today = date.today()
                    next_satday = today + rdelta.relativedelta(days=1, weekday=rdelta.SA(+1))
                    entity = datetime.datetime.strftime(next_satday, "%d-%m-%Y")
                    Date.append(entity)
    
                elif(tag0[i]=='sun' or tag0[i]=='Sun' or tag0[i]=='SUN' or tag0[i]=='sunday' or tag0[i]=='Sunday' or tag0[i]=='SUNDAY'):
                    today = date.today()
                    next_sunday = today + rdelta.relativedelta(days=1, weekday=rdelta.SU(+1))
                    entity = datetime.datetime.strftime(next_sunday, "%d-%m-%Y")
                    Date.append(entity)
    
            #last monday -> tuesday -> wednesday -> thursday -> friday -> saturday -> sunday
            if(tag0[i-1]=="last" or tag0[i-1]=="Last" or tag0[i-1]=="LAST"):
                if(tag0[i]=='mon' or tag0[i]=='Mon' or tag0[i]=='MON' or tag0[i]=='monday' or tag0[i]=='Monday' or tag0[i]=='MONDAY'):
                    today = date.today()
                    last_monday = today + rdelta.relativedelta(days=-1, weekday=rdelta.MO(-1))
                    entity = datetime.datetime.strftime(last_monday, "%d-%m-%Y")
                    Date.append(entity)
    
                elif(tag0[i]=='tue' or tag0[i]=='Tue' or tag0[i]=='TUE' or tag0[i]=='tues' or tag0[i]=='Tues' or tag0[i]=='TUES' or tag0[i]=='tuesday' or tag0[i]=='Tuesday' or tag0[i]=='TUESDAY'):
                    today = date.today()
                    last_tueday = today + rdelta.relativedelta(days=-1, weekday=rdelta.TU(-1))
                    entity = datetime.datetime.strftime(last_tueday, "%d-%m-%Y")
                    Date.append(entity)
    
                elif(tag0[i]=='wed' or tag0[i]=='Wed' or tag0[i]=='WED' or tag0[i]=='wednesday' or tag0[i]=='Wednesday' or tag0[i]=='WEDNESDAY'):
                    today = date.today()
                    weekday = today.weekday()
                    last_wedday = today + rdelta.relativedelta(days=-1, weekday=rdelta.WE(-1))
                    entity = datetime.datetime.strftime(last_wedday, "%d-%m-%Y")
                    Date.append(entity)
    
                elif(tag0[i]=='thu' or tag0[i]=='Thu' or tag0[i]=='THU' or tag0[i]=='thurs' or tag0[i]=='Thurs' or tag0[i]=='THURS' or tag0[i]=='thursday' or tag0[i]=='Thursday' or tag0[i]=='THURSDAY'):
                    today = date.today()
                    last_thuday = today + rdelta.relativedelta(days=-1, weekday=rdelta.TH(-1))
                    entity = datetime.datetime.strftime(last_thuday, "%d-%m-%Y")
                    Date.append(entity)
    
                elif(tag0[i]=='fri' or tag0[i]=='Fri' or tag0[i]=='FRI' or tag0[i]=='friday' or tag0[i]=='Friday' or tag0[i]=='FRIDAY'):
                    today = date.today()
                    last_friday = today + rdelta.relativedelta(days=-1, weekday=rdelta.FR(-1))
                    entity = datetime.datetime.strftime(last_friday, "%d-%m-%Y")
                    Date.append(entity)
    
                elif(tag0[i]=='sat' or tag0[i]=='Sat' or tag0[i]=='SAT' or tag0[i]=='saturday' or tag0[i+1]=='Saturday' or tag0[i]=='SATURDAY'):
                    today = date.today()
                    last_satday = today + rdelta.relativedelta(days=-1, weekday=rdelta.SA(-1))
                    entity = datetime.datetime.strftime(last_satday, "%d-%m-%Y")
                    Date.append(entity)
    
                elif(tag0[i]=='sun' or tag0[i]=='Sun' or tag0[i]=='SUN' or tag0[i]=='sunday' or tag0[i]=='Sunday' or tag0[i]=='SUNDAY'):
                    today = date.today()
                    last_sunday = today + rdelta.relativedelta(days=-1, weekday=rdelta.SU(-1))
                    entity = datetime.datetime.strftime(last_sunday, "%d-%m-%Y")
                    Date.append(entity)
                    
        i=i+1
    print("We will write to csv "+str(Date))  
