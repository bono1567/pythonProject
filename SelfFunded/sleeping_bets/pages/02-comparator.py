import solara
import os
print(os.getcwd())
import numpy as np
import ScraperFC as sfc
from analytics import PlayerAnalytics

ss = sfc.Sofascore()
np.seterr(all='ignore')
seasons = ["24/25", "23/24", "22/23"]
all_columns = ['goals', 'yellowCards', 'redCards', 'groundDuelsWon',
       'groundDuelsWonPercentage', 'aerialDuelsWon',
       'aerialDuelsWonPercentage', 'successfulDribbles',
       'successfulDribblesPercentage', 'tackles', 'assists',
       'accuratePassesPercentage', 'totalDuelsWon', 'totalDuelsWonPercentage',
       'minutesPlayed', 'wasFouled', 'fouls', 'dispossessed', 'appearances',
       'saves', 'savedShotsFromInsideTheBox', 'savedShotsFromOutsideTheBox',
       'goalsConcededInsideTheBox', 'goalsConcededOutsideTheBox', 'accurateFinalThirdPasses',
       'bigChancesCreated', 'accuratePasses', 'keyPasses', 'accurateCrosses',
       'accurateCrossesPercentage', 'accurateLongBalls',
       'accurateLongBallsPercentage', 'interceptions', 'clearances',
       'dribbledPast', 'bigChancesMissed', 'totalShots', 'shotsOnTarget',
       'blockedShots', 'goalConversionPercentage', 'hitWoodwork', 'offsides',
       'expectedGoals', 'errorLeadToGoal', 'errorLeadToShot', 'passToAssist',
       'player', 'team', 'player id', 'team id']

@solara.component
def Page():
    solara.lab.ThemeToggle()
    solara.Title("Sleeping Bets")
    solara.Markdown("#😴 Sleeping Bets 😴")
    with solara.Card("Comparison Page"):
        comp_selected_players = solara.reactive("Kevin De Bruyne,Diogo Jota")
        comp_season = solara.reactive("24/25")
        solara.InputText("Enter comma sperate player names to compare", 
                         value=comp_selected_players)
        solara.Select(label="Season", values=seasons, value=comp_season)
        solara.Markdown(comp_selected_players.value)
        comp_feature = solara.reactive("goals")
        solara.Select(label="Select Feature", values=all_columns, value=comp_feature)
    
        def player_analytics():
            print("Running player analytics")
            player_analysis = PlayerAnalytics([x.strip() for x in comp_selected_players.value.split(",")], sfc=ss, season=comp_season.value)
            comp_historic_sim_data = solara.reactive(player_analysis.get_historic_similarity_stats())
            plt = player_analysis.plot_comparison(comp_feature.value, comp_historic_sim_data.value)
            solara.display(plt.show())

        solara.Button("Run Analysis", on_click=player_analytics)
            
Page()