# Figuras antigas

Guardadas só para referência. **Não use no relatório**: várias foram desenhadas a
partir de rodadas de K-means sem semente fixa, então as cores e os números dos
estados não batem com a partição de referência (`canonical_states`) usada no
texto.

As figuras do relatório estão em [`../relatorio/`](../relatorio/LEGENDAS.md) e
são regeneradas por `python -m molsim.figuras`.

| antiga | substituída por | por quê |
|---|---|---|
| `elbow_rg_rmsd.png` | `A_kmeans_rg_rmsd` (a) | sem código no repositório que a gere |
| `kmeans_rg_rmsd.png` | `A_kmeans_rg_rmsd` (b) | sem código; 4 cores em dispersão não se distinguem |
| `ramachandran_single.png` | `E_ramachandran_estados`, coluna Ala3 | rótulos de estado sem semente |
| `ramachandran_grid.png` | `E_ramachandran_estados` | rótulos de estado sem semente |
| `pca_clusters.png` | `E_dpca_estados` | rótulos sem semente; PCA nos ângulos crus ignora a periodicidade |
| `exemplo_media_circular.png` | `B_media_circular` | sem código no repositório |
| `convergencia_variantes.png` | `B_convergencia` | mesmos números, agora gerada por código |
| `ari_baselines.png` | `B_ari_metodos` | mesmos números, agora gerada por código |
| `cotovelo_corda.png` | `B_cotovelo_corda` | versão do notebook 05 |
| `sementes_k4.png` | `B_sementes` | versão do notebook 05 |
| `estabilidade_estados.png` | `E_estabilidade` (b) | versão do notebook 05 |
| `ramachandran_estados.png` | `E_ramachandran_estados` | versão do notebook 05 |
| `dpca_estados.png` | `E_dpca_estados` | versão do notebook 05 |
| `residencia_k4.png` | `C_residencia` | versão do notebook 05 |
| `its_k4.png`, `its_microestados.png` | `C_its` | as duas numa figura só |
| `macroestados_alfaL.png` | `C_macroestados_alfaL` | versão do notebook 05 |
| `rotulos_tempo.png` | `C_rotulos_tempo` | versão do notebook 05 |
