import urllib

seasons_1 = list(range(93,118,1))

for i in range(len(seasons_1)):
    seasons_1[i] = str(seasons_1[i])
    while len(seasons_1[i]) > 2:
        seasons_1[i] = seasons_1[i][-2:]
    else:
        continue

seasons_2 = list(range(94,119,1))

for i in range(len(seasons_2)):
    seasons_2[i] = str(seasons_2[i])
    while len(seasons_2[i]) > 2:
        seasons_2[i] = seasons_2[i][-2:]
    else:
        continue

seasons = [seasons_1[i] + seasons_2[i] for i in range(0,len(seasons_1))]

url_list = []
filename_list = []

for i in range(len(seasons)):
    url_list.append('https://www.football-data.co.uk/mmz4281/' + seasons[i] + '/E0.csv')

for i in range(len(seasons)):
    filename_list.append('premierleague_' + seasons[i])

for i in range(len(seasons)):
    downloadfile = urllib.URLopener()
    downloadfile.retrieve(url_list[i], filename_list[i])