import csv
import os
import nltk
import datetime
from datetime import date
from datetime import timedelta
from nltk import RegexpTagger
from nltk import word_tokenize, pos_tag

file_name, file_extension = os.path.splitext("test.csv")


if(file_extension == ".csv"):
    with open("test.csv", "r") as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            text=row

#Making some pattern to match every token
patterns = [
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  #cardinal numbers
    (r'(sunday|SUNDAY|Sunday|sun|SUN|Sun|Monday|MONDAY|monday|Mon|MON|mon|Tuesday|TUESDAY|tuesday|Tues|TUES|tues|Wednesday|WEDNESDAY|wednesday|Wed|WED|wed|Thursday|THURSDAY|thursday|Thurs|thurs|THURS|Friday|FRIDAY|friday|fri|Fri|FRI|Saturday|SATURDAY|saturday|Sat|sat|SAT)$','DAYS'),
    (r'(January|january|JANUARY|February|february|FEBRUARY|March|MARCH|march|April|APRIL|april|May|may|MAY|June|june|JUNE|JULY|July|july|August|AUGUST|august|September|Sep|SEPTEMBER|september|October|OCTOBER|october|November|NOVEMBER|november|December|DECEMBER|december)$','MONTH'), #month
    (r'(JAN|jan|Jan|Feb|FEB|feb|Mar|MAR|mar|Apr|apr|APR|May|may|MAY|JUN||jun|Jun|jul|JUL|Jul|Aug|aug|AUG|Sep|sep|SEP|oct|Oct|OCT|Nov|nov|NOV|Dec|dec|DEC)$','month'), #month
    (r'\d{2}/\d{2}/\d{4}', 'DATE'),   #Date
    (r'\d{2}/\d{2}/\d{2}', 'DATE'),
    (r'\d{1}/\d{2}/\d{2}', 'DATE'),
    (r'\d{1}/\d{2}/\d{4}', 'DATE'),
    (r'\d{2}/\d{1}/\d{2}', 'DATE'),
    (r'\d{1}/\d{1}/\d{4}', 'DATE'),
    (r'\d{1}/\d{1}/\d{2}', 'DATE'),
    (r'\d{2}/\d{1}/\d{4}', 'DATE'),   
    (r'\d{2}.\d{2}.\d{4}', 'DATE'),   
    (r'\d{2}.\d{2}.\d{2}', 'DATE'),
    (r'\d{1}.\d{2}.\d{2}', 'DATE'),
    (r'\d{1}.\d{2}.\d{4}', 'DATE'),
    (r'\d{2}.\d{1}.\d{2}', 'DATE'),
    (r'\d{2}.\d{1}.\d{4}', 'DATE'),
    (r'\d{1}.\d{1}.\d{4}', 'DATE'),
    (r'\d{1}.\d{1}.\d{2}', 'DATE'),   
    (r'\d{2}-\d{2}-\d{4}', 'DATE'),   
    (r'\d{2}-\d{2}-\d{2}', 'DATE'),
    (r'\d{1}-\d{2}-\d{2}', 'DATE'),
    (r'\d{1}-\d{2}-\d{4}', 'DATE'),
    (r'\d{2}-\d{1}-\d{2}', 'DATE'),
    (r'\d{2}-\d{1}-\d{4}', 'DATE'),
    (r'\d{1}-\d{1}-\d{2}', 'DATE'),
    (r'\d{1}-\d{1}-\d{4}', 'DATE'),
    (r'\d{1}-(January|january|JANUARY|Jan|jan|JAN|February|february|FEBRUARY|Feb|feb|FEB|March|march|MARCH|Mar|mar|MAR|April|april|APRIL|Apr|apr|APR|May|may|MAY|June|june|JUNE|Jun|jun|JUN|July|july|JULY|Jul|jul|JUL|August|august|AUGUST|Aug|aug|AUG|September|september|SEPTEMBER|Sep|sep|SEP|October|october|OCTOBER|Oct|oct|OCT|November|november|NOVEMBER|Nov|nov|NOV|December|december|DECEMBER|Dec|dec|DEC)-\d{2}', 'DATE_1_2'),
    (r'\d{1}-(January|january|JANUARY|Jan|jan|JAN|February|february|FEBRUARY|Feb|feb|FEB|March|march|MARCH|Mar|mar|MAR|April|april|APRIL|Apr|apr|APR|May|may|MAY|June|june|JUNE|Jun|jun|JUN|July|july|JULY|Jul|jul|JUL|August|august|AUGUST|Aug|aug|AUG|September|september|SEPTEMBER|Sep|sep|SEP|October|october|OCTOBER|Oct|oct|OCT|November|november|NOVEMBER|Nov|nov|NOV|December|december|DECEMBER|Dec|dec|DEC)-\d{4}', 'DATE_1_4'),
    (r'\d{2}-(January|january|JANUARY|Jan|jan|JAN|February|february|FEBRUARY|Feb|feb|FEB|March|march|MARCH|Mar|mar|MAR|April|april|APRIL|Apr|apr|APR|May|may|MAY|June|june|JUNE|Jun|jun|JUN|July|july|JULY|Jul|jul|JUL|August|august|AUGUST|Aug|aug|AUG|September|september|SEPTEMBER|Sep|sep|SEP|October|october|OCTOBER|Oct|oct|OCT|November|november|NOVEMBER|Nov|nov|NOV|December|december|DECEMBER|Dec|dec|DEC)-\d{2}', 'DATE_2_2'),
    (r'\d{2}-(January|january|JANUARY|Jan|jan|JAN|February|february|FEBRUARY|Feb|feb|FEB|March|march|MARCH|Mar|mar|MAR|April|april|APRIL|Apr|apr|APR|May|may|MAY|June|june|JUNE|Jun|jun|JUN|July|july|JULY|Jul|jul|JUL|August|august|AUGUST|Aug|aug|AUG|September|september|SEPTEMBER|Sep|sep|SEP|October|october|OCTOBER|Oct|oct|OCT|November|november|NOVEMBER|Nov|nov|NOV|December|december|DECEMBER|Dec|dec|DEC)-\d{4}', 'DATE_2_4'),
    (r'(1st|2nd|3rd|4th|5th|6th|7th|8th|9th|10th|11th|12th|13th|14th|15th|16th|17th|18th|19th|20th|21st|22nd|23rd|24th|25th|26th|27th|28th|29th|30th|31st)$', 'DATEUSP'), #DATEUSP
    (r'.*', 'NN') # nouns (default)
    ]
 
reg_tagger = RegexpTagger(patterns) 
#tokens = word_tokenize(text)      #Tokenization of every word of a sentence

tagged = reg_tagger.tag(text)      #adding parts of speech tag to each text word
#tagged = reg_tagger.tag(tokens)   #adding parts of speech tag to each token

#print(tagged)

tag0 = [e[0] for e in tagged]     #gaining the first value of a list

tag1 = [e[1] for e in tagged]     #gaining the second value of a list

#print(tag0)
#print(tag1)

i=0
list_len = len(tag1) #get the total length of the list
while(i < list_len):
    if(tag1[i]=="DATE"):
        print(tag0[i])

    #for 21-january-2018 or 21-jan-18
    if(tag1[i]=="DATE_1_2" or tag1[i]=="DATE_1_4" or tag1[i]=="DATE_2_2" or tag1[i]=="DATE_2_4"):
        data = tag0[i]
        data1 = data.split('-')
        if(data1[1]=="January" or data1[1]=="january" or data1[1]=="JANUARY" or data1[1]=="Jan" or data1[1]=="jan" or data1[1]=="JAN"):
            data1[1]=1
            day = str(data1[0])
            month = str(data1[1])
            year = str(data1[2])
            print(day+"-"+month+"-"+year)
        elif(data1[1]=="February" or data1[1]=="february" or data1[1]=="FEBRUARY" or data1[1]=="Feb" or data1[1]=="feb" or data1[1]=="FEB"):
            data1[1]=2
            day = str(data1[0])
            month = str(data1[1])
            year = str(data1[2])
            print (day+"-"+month+"-"+year)      
        elif(data1[1]=="March" or data1[1]=="march" or data1[1]=="MARCH" or data1[1]=="Mar" or data1[1]=="mar" or data1[1]=="MAR"):
            data1[1]=3
            day = str(data1[0])
            month = str(data1[1])
            year = str(data1[2])
            print (day+"-"+month+"-"+year) 
        elif(data1[1]=="April" or data1[1]=="april" or data1[1]=="APRIL" or data1[1]=="Apr" or data1[1]=="apr" or data1[1]=="APR"):
            data1[1]=4
            day = str(data1[0])
            month = str(data1[1])
            year = str(data1[2])
            print (day+"-"+month+"-"+year) 
        elif(data1[1]=="May" or data1[1]=="may" or data1[1]=="MAY"):
            data1[1]=5
            day = str(data1[0])
            month = str(data1[1])
            year = str(data1[2])
            print (day+"-"+month+"-"+year) 
        elif(data1[1]=="June" or data1[1]=="june" or data1[1]=="JUNE" or data1[1]=="Jun" or data1[1]=="jun" or data1[1]=="JUN"):
            data1[1]=6
            day = str(data1[0])
            month = str(data1[1])
            year = str(data1[2])
            print (day+"-"+month+"-"+year) 
        elif(data1[1]=="July" or data1[1]=="july" or data1[1]=="JULY" or data1[1]=="Jul" or data1[1]=="jul" or data1[1]=="JUL"):
            data1[1]=7
            day = str(data1[0])
            month = str(data1[1])
            year = str(data1[2])
            print (day+"-"+month+"-"+year) 
        elif(data1[1]=="August" or data1[1]=="august" or data1[1]=="AUGUST" or data1[1]=="Aug" or data1[1]=="aug" or data1[1]=="AUG"):
            data1[1]=8
            day = str(data1[0])
            month = str(data1[1])
            year = str(data1[2])
            print (day+"-"+month+"-"+year) 
        elif(data1[1]=="September" or data1[1]=="september" or data1[1]=="SEPTEMBER" or data1[1]=="Sep" or data1[1]=="sep" or data1[1]=="SEP"):
            data1[1]=9
            day = str(data1[0])
            month = str(data1[1])
            year = str(data1[2])
            print (day+"-"+month+"-"+year) 
        elif(data1[1]=="October" or data1[1]=="october" or data1[1]=="OCTOBER" or data1[1]=="Oct" or data1[1]=="oct" or data1[1]=="OCT"):
            data1[1]=10
            day = str(data1[0])
            month = str(data1[1])
            year = str(data1[2])
            print (day+"-"+month+"-"+year) 
        elif(data1[1]=="November" or data1[1]=="november" or data1[1]=="NOVEMBER" or data1[1]=="Nov" or data1[1]=="nov" or data1[1]=="NOV"):
            data1[1]=11
            day = str(data1[0])
            month = str(data1[1])
            year = str(data1[2])
            print (day+"-"+month+"-"+year) 
        elif(data1[1]=="December" or data1[1]=="december" or data1[1]=="DECEMBER" or data1[1]=="Dec" or data1[1]=="dec" or data1[1]=="DEC"):
            data1[1]=12
            day = str(data1[0])
            month = str(data1[1])
            year = str(data1[2])
            print (day+"-"+month+"-"+year)
            

    #for 21 january 2018 and 21 jan 2018
    if(tag1[i-1]=="CD"):
        if(tag1[i]=="MONTH"):
            date_formate = tag0[i-1]+" "+tag0[i]+" "+tag0[i+1]
            if(len(tag0[i+1]) == 2):
                date_time_obj = datetime.datetime.strptime(date_formate, '%d %B %y')
                ab= date_time_obj.date()
                print (datetime.datetime.strftime(ab, "%d-%m-%Y"))
            elif(len(tag0[i+1]) == 4):
                date_time_obj = datetime.datetime.strptime(date_formate, '%d %B %Y')
                ab= date_time_obj.date()
                print (datetime.datetime.strftime(ab, "%d-%m-%Y"))
            
        if(tag1[i]=="month"):
            date_formate = tag0[i-1]+" "+tag0[i]+" "+tag0[i+1]
            if(len(tag0[i+1]) == 2):
                date_time_obj = datetime.datetime.strptime(date_formate, '%d %b %y')
                ab= date_time_obj.date()
                print (datetime.datetime.strftime(ab, "%d-%m-%Y"))
            elif(len(tag0[i+1]) == 4):
                date_time_obj = datetime.datetime.strptime(date_formate, '%d %b %Y')
                ab= date_time_obj.date()
                print (datetime.datetime.strftime(ab, "%d-%m-%Y"))

      
    if(tag1[i-1]=="DATEUSP"):
        #for 21st january 2018 and 21st jan 2018
        if(tag0[i-1].endswith("st")==True):
            if(tag1[i]=="MONTH"):
                date_formate = tag0[i-1]+" "+tag0[i]+" "+tag0[i+1]
                if(len(tag0[i+1]) == 2):
                    date_time_obj = datetime.datetime.strptime(date_formate, '%dst %B %y')
                    ab= date_time_obj.date()
                    print (datetime.datetime.strftime(ab, "%d-%m-%Y"))
                elif(len(tag0[i+1]) == 4):
                    date_time_obj = datetime.datetime.strptime(date_formate, '%dst %B %Y')
                    ab= date_time_obj.date()
                    print (datetime.datetime.strftime(ab, "%d-%m-%Y"))
            if(tag1[i]=="month"):
                date_formate = tag0[i-1]+" "+tag0[i]+" "+tag0[i+1]
                if(len(tag0[i+1]) == 2):
                    date_time_obj = datetime.datetime.strptime(date_formate, '%dst %b %y')
                    ab= date_time_obj.date()
                    print (datetime.datetime.strftime(ab, "%d-%m-%Y"))
                elif(len(tag0[i+1]) == 4):
                    date_time_obj = datetime.datetime.strptime(date_formate, '%dst %b %Y')
                    ab= date_time_obj.date()
                    print (datetime.datetime.strftime(ab, "%d-%m-%Y"))
        
        #for 22nd january 2018 and 22nd jan 2018
        elif(tag0[i-1].endswith("nd")==True):
            if(tag1[i]=="MONTH"):
                date_formate = tag0[i-1]+" "+tag0[i]+" "+tag0[i+1]
                if(len(tag0[i+1]) == 2):
                    date_time_obj = datetime.datetime.strptime(date_formate, '%dnd %B %y')
                    ab= date_time_obj.date()
                    print (datetime.datetime.strftime(ab, "%d-%m-%Y"))
                elif(len(tag0[i+1]) == 4):
                    date_time_obj = datetime.datetime.strptime(date_formate, '%dnd %B %Y')
                    ab= date_time_obj.date()
                    print (datetime.datetime.strftime(ab, "%d-%m-%Y"))
            if(tag1[i]=="month"):
                date_formate = tag0[i-1]+" "+tag0[i]+" "+tag0[i+1]
                if(len(tag0[i+1]) == 2):
                    date_time_obj = datetime.datetime.strptime(date_formate, '%dnd %b %y')
                    ab= date_time_obj.date()
                    print (datetime.datetime.strftime(ab, "%d-%m-%Y"))
                elif(len(tag0[i+1]) == 4):
                    date_time_obj = datetime.datetime.strptime(date_formate, '%dnd %b %Y')
                    ab= date_time_obj.date()
                    print (datetime.datetime.strftime(ab, "%d-%m-%Y"))
        
        #for 23rd january 2018 and 23rd jan 2018
        elif(tag0[i-1].endswith("rd")==True):
            if(tag1[i]=="MONTH"):
                date_formate = tag0[i-1]+" "+tag0[i]+" "+tag0[i+1]
                if(len(tag0[i+1]) == 2):
                    date_time_obj = datetime.datetime.strptime(date_formate, '%drd %B %y')
                    ab= date_time_obj.date()
                    print (datetime.datetime.strftime(ab, "%d-%m-%Y"))
                elif(len(tag0[i+1]) == 4):
                    date_time_obj = datetime.datetime.strptime(date_formate, '%drd %B %Y')
                    ab= date_time_obj.date()
                    print (datetime.datetime.strftime(ab, "%d-%m-%Y"))
            if(tag1[i]=="month"):
                date_formate = tag0[i-1]+" "+tag0[i]+" "+tag0[i+1]
                if(len(tag0[i+1]) == 2):
                    date_time_obj = datetime.datetime.strptime(date_formate, '%drd %b %y')
                    ab= date_time_obj.date()
                    print (datetime.datetime.strftime(ab, "%d-%m-%Y"))
                elif(len(tag0[i+1]) == 4):
                    date_time_obj = datetime.datetime.strptime(date_formate, '%drd %b %Y')
                    ab= date_time_obj.date()
                    print (datetime.datetime.strftime(ab, "%d-%m-%Y"))
        
        #for 24th january 2018 and 24th jan 2018
        elif(tag0[i-1].endswith("th")==True):
            if(tag1[i]=="MONTH"):
                date_formate = tag0[i-1]+" "+tag0[i]+" "+tag0[i+1]
                if(len(tag0[i+1]) == 2):
                    date_time_obj = datetime.datetime.strptime(date_formate, '%dth %B %y')
                    ab= date_time_obj.date()
                    print (datetime.datetime.strftime(ab, "%d-%m-%Y"))
                elif(len(tag0[i+1]) == 4):
                    date_time_obj = datetime.datetime.strptime(date_formate, '%dth %B %Y')
                    ab= date_time_obj.date()
                    print (datetime.datetime.strftime(ab, "%d-%m-%Y"))
            if(tag1[i]=="month"):
                date_formate = tag0[i-1]+" "+tag0[i]+" "+tag0[i+1]
                if(len(tag0[i+1]) == 2):
                    date_time_obj = datetime.datetime.strptime(date_formate, '%dth %b %y')
                    ab= date_time_obj.date()
                    print (datetime.datetime.strftime(ab, "%d-%m-%Y"))
                elif(len(tag0[i+1]) == 4):
                    date_time_obj = datetime.datetime.strptime(date_formate, '%dth %b %Y')
                    ab= date_time_obj.date()
                    print (datetime.datetime.strftime(ab, "%d-%m-%Y"))

    #last/privous year -> year -> month -> week
    if(tag0[i]=="last" or tag0[i]=="Last" or tag0[i]=="LAST" or tag0[i]=="previous" or tag0[i]=="Previous" or tag0[i]=="PREVIOUS"):
        if(tag0[i+1]=="year" or tag0[i+1]=="Year" or tag0[i+1]=="YEAR"):
            now = datetime.datetime.now()
            now_to_last_year = now.year - 1
            last_year = str(now_to_last_year)
            print ("01-01-"+last_year)
        elif(tag0[i+1]=="month" or tag0[i+1]=="Month" or tag0[i+1]=="MONTH"):
            now = datetime.datetime.now()
            this_year = str(now.year)
            now_to_last_month = (now.month - 1)%12
            last_month = str(now_to_last_month)
            print ("01-"+last_month+"-"+this_year)
        elif(tag0[i+1]=="week" or tag0[i+1]=="Week" or tag0[i+1]=="WEEK"):
            today = datetime.datetime.today()
            next_week = today-datetime.timedelta(weeks=1)
            print (datetime.datetime.strftime(next_week, "%d-%m-%Y"))
            
    #this year -> this month -> this week
    if(tag0[i]=="This" or tag0[i]=="this" or tag0[i]=="THIS"):
        if(tag0[i+1]=="year" or tag0[i+1]=="Year" or tag0[i+1]=="YEAR"):
            now = datetime.datetime.now()
            this_year = str(now.year)
            print ("01-01-"+this_year)
        elif(tag0[i+1]=="month" or tag0[i+1]=="Month" or tag0[i+1]=="MONTH"):
            now = datetime.datetime.now()
            pre_month = str(now.month)
            pre_year = str(now.year)
            print ("01-"+pre_month+"-"+pre_year)
        elif(tag0[i+1]=='week' or tag0[i+1]=='Week' or tag0[i+1]=='WEEK'):
            print (datetime.datetime.now().strftime("%d-%m-%Y"))

    #next year -> month -> week
    if(tag0[i]=="next" or tag0[i]=="Next" or tag0[i]=="NEXT"):
        if(tag0[i+1]=="year" or tag0[i+1]=="Year" or tag0[i+1]=="YEAR"):
            now = datetime.datetime.now()
            now_to_next_year = now.year + 1
            next_year = str(now_to_next_year)
            print ("01-01-"+next_year)
        elif(tag0[i+1]=="month" or tag0[i+1]=="Month" or tag0[i+1]=="MONTH"):
            now = datetime.datetime.now()
            this_year = now.year
            now_to_next_month = now.month + 1
            if(now_to_next_month > 12):
                next_month = now_to_next_month % 12
                year = this_year + 1
                print ("01-"+str(next_month)+"-"+str(year))
            else:
                next_month = str(now_to_next_month)
                print ("01-"+str(now_to_next_month)+"-"+str(this_year))
        elif(tag0[i+1]=="week" or tag0[i+1]=="Week" or tag0[i+1]=="WEEK"):
            today = datetime.datetime.today()
            next_week = today+datetime.timedelta(weeks=1)
            print (datetime.datetime.strftime(next_week, "%d-%m-%Y"))
        
    #today   
    if(tag0[i]=='today' or tag0[i]=='Today' or tag0[i]=='TODAY' or 
    (tag0[i]=="present" and tag0[i+1]=="day") or (tag0[i]=="Present" and tag0[i+1]=="day") or 
    (tag0[i]=="Present" and tag0[i+1]=="Day") or 
    (tag0[i]=="PRESENT" and tag0[i+1]=="DAY")):
        print (datetime.datetime.now().strftime("%d-%m-%Y"))

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
        print (datetime.datetime.strftime(yesterday, "%d-%m-%Y"))
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
            print (datetime.datetime.strftime(tomorrow, "%d-%m-%Y"))

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
        print (datetime.datetime.strftime(yesterday, "%d-%m-%Y"))

    if(tag1[i]=='DAYS'):   
        #next monday -> tuesday -> wednesday -> thursday -> friday -> saturday -> sunday
        if(tag0[i-1]=='next' or tag0[i-1]=="Next" or tag0[i-1]=="NEXT"):
            if(tag0[i]=='mon' or tag0[i]=='Mon' or tag0[i]=='MON' or tag0[i]=='monday' or tag0[i]=='Monday' or tag0[i]=='MONDAY'):
                today = date.today()
                weekday = today.weekday()
                offset = (today.weekday()) % 7
                if(offset == weekday):
                    next_monday = today + timedelta(weeks=1)
                    print (datetime.datetime.strftime(next_monday, "%d-%m-%Y"))
                else:
                    next_monday = today + timedelta(days=offset)
                    print (datetime.datetime.strftime(next_monday, "%d-%m-%Y"))

            elif(tag0[i]=='tue' or tag0[i]=='Tue' or tag0[i]=='TUE' or tag0[i]=='tues' or tag0[i]=='Tues' or tag0[i]=='TUES' or tag0[i]=='tuesday' or tag0[i]=='Tuesday' or tag0[i]=='TUESDAY'):
                today = date.today()
                weekday = today.weekday()
                offset = (today.weekday()+1) % 7
                if(offset == weekday):
                    next_tuesday = today + timedelta(weeks=1)
                    print (datetime.datetime.strftime(next_tuesday, "%d-%m-%Y"))
                else:
                    next_tuesday = today + timedelta(days=offset)
                    print (datetime.datetime.strftime(next_tuesday, "%d-%m-%Y"))

            elif(tag0[i]=='wed' or tag0[i]=='Wed' or tag0[i]=='WED' or tag0[i]=='wednesday' or tag0[i]=='Wednesday' or tag0[i]=='WEDNESDAY'):
                today = date.today()
                weekday = today.weekday()
                offset = (today.weekday()+2) % 7
                if(offset == weekday):
                    next_wednesday = today + timedelta(weeks=1)
                    print (datetime.datetime.strftime(next_wednesday, "%d-%m-%Y"))
                else:
                    next_wednesday = today + timedelta(days=offset)
                    print (datetime.datetime.strftime(next_wednesday, "%d-%m-%Y"))

            elif(tag0[i]=='thu' or tag0[i]=='Thu' or tag0[i]=='THU' or tag0[i]=='thurs' or tag0[i]=='Thurs' or tag0[i]=='THURS' or tag0[i]=='thursday' or tag0[i]=='Thursday' or tag0[i]=='THURSDAY'):
                today = date.today()
                weekday = today.weekday()
                offset = (today.weekday()+3) % 7
                if(offset == weekday):
                    next_thursday = today + timedelta(weeks=1)
                    print (datetime.datetime.strftime(next_thursday, "%d-%m-%Y"))
                else:
                    next_thursday = today + timedelta(days=offset)
                    print (datetime.datetime.strftime(next_thursday, "%d-%m-%Y"))

            elif(tag0[i]=='fri' or tag0[i]=='Fri' or tag0[i]=='FRI' or tag0[i]=='friday' or tag0[i]=='Friday' or tag0[i]=='FRIDAY'):
                today = date.today()
                weekday = today.weekday()
                offset = (today.weekday()+4) % 7
                if(offset == weekday):
                    next_friday = today + timedelta(weeks=1)
                    print (datetime.datetime.strftime(next_friday, "%d-%m-%Y"))
                else:
                    next_friday = today + timedelta(days=offset)
                    print (datetime.datetime.strftime(next_friday, "%d-%m-%Y"))

            elif(tag0[i]=='sat' or tag0[i]=='Sat' or tag0[i]=='SAT' or tag0[i]=='saturday' or tag0[i+1]=='Saturday' or tag0[i]=='SATURDAY'):
                today = date.today()
                weekday = today.weekday()
                offset = (today.weekday()+5) % 7
                if(offset == weekday):
                    next_saturday = today + timedelta(weeks=1)
                    print (datetime.datetime.strftime(next_saturday, "%d-%m-%Y"))
                else:
                    next_saturday = today + timedelta(days=offset)
                    print (datetime.datetime.strftime(next_saturday, "%d-%m-%Y"))

            elif(tag0[i]=='sun' or tag0[i]=='Sun' or tag0[i]=='SUN' or tag0[i]=='sunday' or tag0[i]=='Sunday' or tag0[i]=='SUNDAY'):
                today = date.today()
                weekday = today.weekday()
                offset = (today.weekday()+6) % 7
                if(offset == weekday):
                    next_sunday = today + timedelta(weeks=1)
                    print (datetime.datetime.strftime(next_sunday, "%d-%m-%Y"))
                else:
                    next_sunday = today + timedelta(days=offset)
                    print (datetime.datetime.strftime(next_sunday, "%d-%m-%Y"))

        #last monday -> tuesday -> wednesday -> thursday -> friday -> saturday -> sunday
        if(tag0[i-1]=="last" or tag0[i-1]=="Last" or tag0[i-1]=="LAST"):
            if(tag0[i]=='mon' or tag0[i]=='Mon' or tag0[i]=='MON' or tag0[i]=='monday' or tag0[i]=='Monday' or tag0[i]=='MONDAY'):
                today = date.today()
                weekday = today.weekday()
                offset = (today.weekday()) % 7
                if(offset == weekday):
                    last_monday = today - timedelta(weeks=1)
                    print (datetime.datetime.strftime(last_monday, "%d-%m-%Y"))
                else:
                    last_monday = today - timedelta(days=offset)
                    print (datetime.datetime.strftime(last_monday, "%d-%m-%Y"))

            elif(tag0[i]=='tue' or tag0[i]=='Tue' or tag0[i]=='TUE' or tag0[i]=='tues' or tag0[i]=='Tues' or tag0[i]=='TUES' or tag0[i]=='tuesday' or tag0[i]=='Tuesday' or tag0[i]=='TUESDAY'):
                today = date.today()
                weekday = today.weekday()
                offset = (today.weekday()+6) % 7
                if(offset == weekday):
                    last_tuesday = today - timedelta(weeks=1)
                    print (datetime.datetime.strftime(last_tuesday, "%d-%m-%Y"))
                else:
                    last_tuesday = today - timedelta(days=offset)
                    print (datetime.datetime.strftime(last_tuesday, "%d-%m-%Y"))

            elif(tag0[i]=='wed' or tag0[i]=='Wed' or tag0[i]=='WED' or tag0[i]=='wednesday' or tag0[i]=='Wednesday' or tag0[i]=='WEDNESDAY'):
                today = date.today()
                weekday = today.weekday()
                offset = (today.weekday()+5) % 7
                if(offset == weekday):
                    last_wednesday = today - timedelta(weeks=1)
                    print (datetime.datetime.strftime(last_wednesday, "%d-%m-%Y"))
                else:
                    last_wednesday = today - timedelta(days=offset)
                    print (datetime.datetime.strftime(last_wednesday, "%d-%m-%Y"))

            elif(tag0[i]=='thu' or tag0[i]=='Thu' or tag0[i]=='THU' or tag0[i]=='thurs' or tag0[i]=='Thurs' or tag0[i]=='THURS' or tag0[i]=='thursday' or tag0[i]=='Thursday' or tag0[i]=='THURSDAY'):
                today = date.today()
                weekday = today.weekday()
                offset = (today.weekday()+4) % 7
                if(offset == weekday):
                    last_thursday = today - timedelta(weeks=1)
                    print (datetime.datetime.strftime(last_thursday, "%d-%m-%Y"))
                else:
                    last_thursday = today - timedelta(days=offset)
                    print (datetime.datetime.strftime(last_thursday, "%d-%m-%Y"))

            elif(tag0[i]=='fri' or tag0[i]=='Fri' or tag0[i]=='FRI' or tag0[i]=='friday' or tag0[i]=='Friday' or tag0[i]=='FRIDAY'):
                today = date.today()
                weekday = today.weekday()
                offset = (today.weekday()+3) % 7
                if(offset == weekday):
                    last_friday = today - timedelta(weeks=1)
                    print (datetime.datetime.strftime(last_friday, "%d-%m-%Y"))
                else:
                    last_friday = today - timedelta(days=offset)
                    print (datetime.datetime.strftime(last_friday, "%d-%m-%Y"))

            elif(tag0[i]=='sat' or tag0[i]=='Sat' or tag0[i]=='SAT' or tag0[i]=='saturday' or tag0[i+1]=='Saturday' or tag0[i]=='SATURDAY'):
                today = date.today()
                weekday = today.weekday()
                offset = (today.weekday()+2) % 7
                if(offset == weekday):
                    last_saturday = today - timedelta(weeks=1)
                    print (datetime.datetime.strftime(last_saturday, "%d-%m-%Y"))
                else:
                    last_saturday = today - timedelta(days=offset)
                    print (datetime.datetime.strftime(last_saturday, "%d-%m-%Y"))

            elif(tag0[i]=='sun' or tag0[i]=='Sun' or tag0[i]=='SUN' or tag0[i]=='sunday' or tag0[i]=='Sunday' or tag0[i]=='SUNDAY'):
                today = date.today()
                weekday = today.weekday()
                offset = (today.weekday()+1) % 7
                if(offset == weekday):
                    last_sunday = today - timedelta(weeks=1)
                    print (datetime.datetime.strftime(last_sunday, "%d-%m-%Y"))
                else:
                    last_sunday = today - timedelta(days=offset)
                    print (datetime.datetime.strftime(last_sunday, "%d-%m-%Y"))

        #this sunday or sunday is the next task to do
        else:
            if(tag0[i]=='mon' or tag0[i]=='Mon' or tag0[i]=='MON' or tag0[i]=='monday' or tag0[i]=='Monday' or tag0[i]=='MONDAY'):
                today = date.today()
                weekday = today.weekday()
                offset = (today.weekday()) % 7
                if(offset == weekday):
                    next_monday = today + timedelta(weeks=0)
                    print (datetime.datetime.strftime(next_monday, "%d-%m-%Y"))
                else:
                    next_monday = today + timedelta(days=offset)
                    print (datetime.datetime.strftime(next_monday, "%d-%m-%Y"))

    #only month name ex-january
    if(tag1[i]=="MONTH"or tag1[i]=="month"):
        if(tag1[i-1]=="DATEUSP" or tag1[i-1]=="CD" and (tag1[i]=="MONTH" or tag1[i]=="month") or tag1[i+1]=="CD" ):
            data = tag0[i-1]+" "+tag0[i]+" "+tag0[i+1]
            #print(data)
        else:
            now = datetime.datetime.now()
            this_year = str(now.year)
            if(tag0[i]=='january' or tag0[i]=='January' or tag0[i]=='jan' or tag0[i]=='Jan'):
                print('01/01/'+this_year)
            elif(tag0[i]=='february' or tag0[i]=='February' or tag0[i]=='feb' or tag0[i]=='Feb'):
                print('01/02/'+this_year)
            elif(tag0[i]=='march' or tag0[i]=='March' or tag0[i]=='mar' or tag0[i]=='Mar'):
                print('01/03/'+this_year)
            elif(tag0[i]=='april' or tag0[i]=='April' or tag0[i]=='apr' or tag0[i]=='Apr'):
                print('01/04/'+this_year)
            elif(tag0[i]=='may' or tag0[i]=='May'):
                print('01/05/'+this_year)
            elif(tag0[i]=='june' or tag0[i]=='June' or tag0[i]=='jun' or tag0[i]=='Jun'):
                print('01/06/'+this_year)
            elif(tag0[i]=='july' or tag0[i]=='July' or tag0[i]=='jul' or tag0[i]=='Jul'):
                print('01/07/'+this_year)
            elif(tag0[i]=='august' or tag0[i]=='August' or tag0[i]=='aug' or tag0[i]=='Aug'):
                print('01/08/'+this_year)
            elif(tag0[i]=='september' or tag0[i]=='September' or tag0[i]=='sep' or tag0[i]=='Sep'):
                print('01/09/'+this_year)
            elif(tag0[i]=='october' or tag0[i]=='October' or tag0[i]=='oct' or tag0[i]=='Oct'):
                print('01/10/'+this_year)
            elif(tag0[i]=='november' or tag0[i]=='November' or tag0[i]=='nov' or tag0[i]=='Nov'):
                print('01/11/'+this_year)
            elif(tag0[i]=='december' or tag0[i]=='December' or tag0[i]=='dec' or tag0[i]=='Dec'):
                print('01/12/'+this_year)
  
    i=i+1