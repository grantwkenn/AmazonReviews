import csv


videoPath = 'amazon_reviews_Video_Games.csv'
kitchenPath = 'amazon_reviews_kitchen.csv'

label = 'video_games'

base_path = 'Amazon Review Datasets/' 

one_star_path = base_path + label + '_one_star.csv'
two_star_path = base_path + label + '_two_star.csv'
three_star_path = base_path + label + '_three_star.csv'
four_star_path = base_path + label + '_four_star.csv'
five_star_path = base_path + label + '_five_star.csv'


file_one = open(one_star_path, 'w')
file_two = open(two_star_path, 'w')
file_three = open(three_star_path, 'w')
file_four = open(four_star_path, 'w')
file_five = open(five_star_path, 'w')

writer_one = csv.writer(file_one) #quotechar="", quoting=csv.QUOTE_NONE)
writer_two = csv.writer(file_two)# quotechar="", quoting=csv.QUOTE_NONE)
writer_three = csv.writer(file_three)# quotechar="", quoting=csv.QUOTE_NONE)
writer_four = csv.writer(file_four)# quotechar="", quoting=csv.QUOTE_NONE)
writer_five = csv.writer(file_five)# quotechar="", quoting=csv.QUOTE_NONE)

def rowToString (row):
    output = ""
    for item in row:
        output += item + ','
    output = output[:-1]
    output += '\n'
    return output


with open(base_path + videoPath) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            row = rowToString(row)
            file_one.write(row)
            file_two.write(row)
            file_three.write(row)
            file_four.write(row)
            file_five.write(row)
            #writer_one.writerow(row)
            #writer_two.writerow(row)
            #writer_three.writerow(row)
            #writer_four.writerow(row)
            #writer_five.writerow(row)
            line_count = 1

        else:
            score = row[7]
            row = rowToString(row)
            print(row)
            if score == str(1):
                #writer_one.writerow(row)
                file_one.write(row)

            elif score == str(2):
                #writer_two.writerow(row)
                file_two.write(row)

            elif score == str(3):
                #writer_three.writerow(row)
                file_three.write(row)
            
            elif score == str(4):
                #writer_four.writerow(row)
                file_four.write(row)

            elif score == str(5):
                #writer_five.writerow(row)
                file_five.write(row)

print('finished')



