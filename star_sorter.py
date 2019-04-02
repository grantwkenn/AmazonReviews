import csv
import numpy
import pandas

videoPath = 'amazon_reviews_Video_Games.csv'
kitchenPath = 'amazon_reviews_kitchen.csv'

label = 'video_games'

base_path = 'Amazon Review Datasets/' 

one_star_path = base_path + label + '_one_star.csv'
two_star_path = base_path + label + '_two_star.csv'
three_star_path = base_path + label + '_three_star.csv'
four_star_path = base_path + label + '_four_star.csv'
five_star_path = base_path + label + '_five_star.csv'


file_one = open(one_star_path, 'w', encoding="utf8")
file_two = open(two_star_path, 'w', encoding="utf8 ")
file_three = open(three_star_path, 'w', encoding="utf8 ")
file_four = open(four_star_path, 'w', encoding="utf8")
file_five = open(five_star_path, 'w', encoding="utf8")

writer_one = csv.writer(file_one, quotechar="", quoting=csv.QUOTE_NONE)
writer_two = csv.writer(file_two, quotechar="", quoting=csv.QUOTE_NONE)
writer_three = csv.writer(file_three, quotechar="", quoting=csv.QUOTE_NONE)
writer_four = csv.writer(file_four, quotechar="", quoting=csv.QUOTE_NONE)
writer_five = csv.writer(file_five, quotechar="", quoting=csv.QUOTE_NONE)

def rowToString (row):
    output = ""
    for item in row:
        output += item + ','
    output = output[:-1]
    output += '\n'
    return output

with open(base_path + videoPath, encoding='utf8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            row = rowToString(row)
            file_one.write(row)
            file_two.write(row)
            file_three.write(row)
            file_four.write(row)
            file_five.write(row)
            line_count = 1

        else:
            score = row[7]
            row = rowToString(row)
            if score == str(1):
                file_one.write(row)

            elif score == str(2):
                file_two.write(row)

            elif score == str(3):
                file_three.write(row)
                
            elif score == str(4):
                file_four.write(row)

            elif score == str(5):
                file_five.write(row)