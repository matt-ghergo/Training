import requests

# We need to download data for the Premier League since their first season until season 17/18
# The links are all identical but for 4 digits indicating the season (eg. the link for season 17/18 will have '1718')
# We will first create a list of strings for each season ['9293', '9394', ...]


# First, we create a list of numbers from 93 until 17
seasons_1 = list(range(93, 118, 1))

# Then, we turn the elements of such a list into strings and we remove the first character when above 100
for i in range(len(seasons_1)):
    seasons_1[i] = str(seasons_1[i])
    while len(seasons_1[i]) > 2:
        seasons_1[i] = seasons_1[i][-2:]
    else:
        continue

# We do the same but for numbers from 94 to 18
seasons_2 = list(range(94, 119, 1))

for i in range(len(seasons_2)):
    seasons_2[i] = str(seasons_2[i])
    while len(seasons_2[i]) > 2:
        seasons_2[i] = seasons_2[i][-2:]
    else:
        continue

# We combine them, to have a list of 4-digit strings each for each season ['9394', '9495', ..., '1718']
seasons = [seasons_1[i] + seasons_2[i] for i in range(0, len(seasons_1))]


# We create a list for the download urls and one for the names we are assigning to the files

url_list = ['https://www.football-data.co.uk/mmz4281/' + seasons[i] + '/E0.csv' for i in range(0, len(seasons))]

filename_list = ['premierleague_' + seasons[i] for i in range(0, len(seasons))]

for i in range(len(seasons)):
    url = url_list[i]
    r = requests.get(url, allow_redirects=True)
    open(filename_list[i], 'wb').write(r.content)
