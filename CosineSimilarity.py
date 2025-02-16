from CosineSimilarityFunctions import similarities

# Calculating similarities takes a considerable amount of time. Therefore, the results of these calculations are saved
# in a directory called MyResultsDir, and they will be retrieved if accessible. If you need to save a new set of
# calculations, you can change this directory to specify a different location for saving.
MyResultsDir = "BridgingGap"
# Is used for versioning new similarity files.
similarities_file_postfix = "250107V1"

# This function runs the similarities' calculations.
similarities(MyResultsDir=MyResultsDir, similarities_file_postfix=similarities_file_postfix, plot_heatmap=True)
