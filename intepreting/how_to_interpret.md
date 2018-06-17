
Experiments:

For every model configuration (2f, 4f, 8f, 64f (8 distinct filters))
   For every word (10 types):
       For 3 words:
           Stap 1: vind de formanten in de audio file
               Open Praat
               Open > Read From File > View & Edit
               Formant > Show Formants
               Write down F1, F2
               Write down speaker sex

               Typ de website in op: https://tophonetics.com/
               Schrijf zowel de Engelse als Amerikaanse versie op, en eventuele variaties (boven transcriptie hoveren met je muis)

               Schrijf de algemene formanten op voor het woord

               Check de uitspraak en formanten per input file.

               Match iedere phone / letter met de plaatjes van Douwe zn boek, om de formant frequenties te achterhalen. Bv yes in de website, geeft ˈjɛs. Match de globale vorm / frequenties van de phone in het plaatje, met die in praat. Dit laatste als bevestiging.

           Stap 2:
               Check which features are activated (check ReLu)
               Omschrijf kort hoe de activatie eruit ziet.
               Omschrijf de activatie frequentie + algemene richting als aanwezig.


Interpretatie:
   For every audio file interpreted in the previous step:
       Compare if features which activate in multiple audio files, with formants which occur in
       multiple audio files with high accuracy

       Do an individual comparison of the feature activation with the formant shape / frequency
