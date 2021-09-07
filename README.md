# README
## Set up & run
1. install requirements
    - > pip install -e ./package
    - > pip install -r requirements.txt  
2. collect data - Optional
    - run collect-udp-metric notebook
    - collect data for dataset
3. run train-gae notebook to create node embeddings 
4. run train-2ornn notebook for learning results
5. run train-extnn notebook for ExtededNN

## TODOS (Typical)
- clean Test and PreCode from notebooks
- ~~make python package~~
- add technical comments e.g. input/output vars description
- set up RPI files structure
- set up all files structure
- fix .gitignore to ignore DataFigures
- ~~set privet epochs counting var (dynamic)~~
- ~~add Extended NN~~
- ~~create BaseModel ?!~~
- ~~modify methods.{print_confmtx, pca_denoising_preprocessing} to be general for other data classes~~
- ~~modify methods.py functions to classes (per comments !?)~~
- ~~!! methods become huge.. any idea to split..~~
- rename thesispack {m2m_deep_pack!?}
- ~~refactor thessispack -> {split models file to base, metrics, models..}~~
- split this repo in two repos (thesis-pack, thesis)
- put "put here a link of repo" thesispack/methods/graphs -> def graph_stats
