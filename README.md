
# League of Legends Chatbot

A chatbot that helps with your queries regarding LOL. This mainly helps the players who are 
trying to set runes just before the game starts, or the ones who wonder which champ to play before a 
game.

To tackle the above, I programmed a model consisting of LSTM RNN which will take care of 
predicting the user's intent and giving the right response for the same. 


## Usage/Examples

Things you can ask the bot:
- Any champ's rune page.
- Any champ's patch history.
- The current Tier list.
- A lore or fact about the game. It can "Surprise" you.
- Current patch notes for the game.
- Get recommendation when you type feeling bored or fun to play champ.
- Broken champs for the current patch and their lane.

## Demo

Live link: https://lol-chatbot.herokuapp.com/


## Run Locally

Clone the project

```bash
  git clone https://github.com/PropaneHD/League-of-Legends-Chat-bot
```

Go to the project directory

```bash
  cd project path
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server

```bash
  streamlit run chatbot_webapp.py
```


## Scope for improvements

- Adding more functionality to the bot like suggesting champs based on their attributes (eg: Tank Support, Assassin Jungle)
- Programming a better model to handle more general queries

## References 

 - For rune pages - https://u.gg/
 - For patch history - https://leagueoflegends.fandom.com/
 - Special thanks - https://discuss.streamlit.io/u/soft-nougat (Background image workaround)



