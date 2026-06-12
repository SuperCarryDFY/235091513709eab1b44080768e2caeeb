ALL_SUBSECTION={'Intramembrane', 'Transit peptide', 'Involvement in disease', 'Signal peptide', 'Binding site', 'Modified residue', 'Chain', 'Tissue specificity',
                'Domain', 'Cross-link', 'Region', 'Proteomes', 'Function', 'Allergenic properties', 'Repeat', 'Miscellaneous', 'Topological domain', 'Glycosylation', 
                'Subunit', 'Toxic dose', 'Active site', 'Subcellular location', 'Caution', 'Developmental stage', 'Domain (non-positional annotation)', 'Compositional bias', 
                'Catalytic activity', 'Virus host', 'Pharmaceutical use', 'Peptide', 'Biophysicochemical properties', 'Pathway', 'Post-translational modification',
                'Induction', 'Mutagenesis', 'Taxonomic lineage', 'Motif', 'Sequence similarities', 'Gene names', 'Activity regulation', 'DNA binding', 'Site', 
                'Cofactor', 'Coiled coil', 'Biotechnology', 'Natural variant', 'Polymorphism', 'Propeptide', 'Disulfide bond', 'Protein names', 'Organism', 
                'Transmembrane', 'Lipidation', 'RNA Editing', 'Disruption phenotype', 'GO annotation'}

residue_level = {"Active site", "Binding site", "Site", "DNA binding", "Natural variant", "Mutagenesis",
                 "Transmembrane", "Topological domain", "Intramembrane", "Signal peptide", "Propeptide",
                 "Transit peptide",
                 "Chain", "Peptide", "Modified residue", "Lipidation", "Glycosylation", "Disulfide bond",
                 "Cross-link",
                 "Domain", "Repeat", "Compositional bias", "Region", "Coiled coil", "Motif"}

sequence_level = {"Function", "Miscellaneous", "Caution", "Catalytic activity", "Cofactor", "Activity regulation",
                  "Biophysicochemical properties", "Pathway", "Involvement in disease", "Allergenic properties",
                  "Toxic dose", "Pharmaceutical use", "Disruption phenotype", "Subcellular location",
                  "Post-translational modification", "Subunit", "Domain (non-positional annotation)",
                  "Sequence similarities", "RNA Editing", "Tissue specificity", "Developmental stage", "Induction",
                  "Biotechnology", "Polymorphism", "GO annotation", "Proteomes", "Protein names", "Gene names",
                  "Organism", "Taxonomic lineage", "Virus host"}

sequence_level2 = {"Function", "Miscellaneous", "Caution", "Catalytic activity", "Cofactor", "Activity regulation",
                  "Biophysicochemical properties", "Pathway", "Involvement in disease", "Allergenic properties",
                  "Toxic dose", "Pharmaceutical use", "Disruption phenotype", "Subcellular location",
                  "Post-translational modification", "Subunit", "Domain (non-positional annotation)",
                  "Sequence similarities", "RNA Editing", "Tissue specificity", "Developmental stage", "Induction",
                  "Biotechnology", "Polymorphism", "GO annotation", "Proteomes", "Protein names", "Gene names",
                  "Organism", "Taxonomic lineage", "Virus host", "Chain"} ## add chain because I think chain can be helpful 
sequence_level3 = {"Taxonomic lineage", "Organism", "Induction", "Tissue specificity", "Subcellular location", "Gene names", 
                    "Post-translational modification", "Subunit", "Disruption phenotype", "GO annotation", "Miscellaneous", 
                    "Function", "Protein names", "Domain (non-positional annotation)", "Sequence similarities", "Catalytic activity", "Pathway"
                    }

SECTION2SUB={
    'Function': ['Function', 'Miscellaneous', 'Caution', 'Catalytic activity', 'Cofactor', 'Activity regulation', 'Biophysicochemical properties', 'Pathway', 'Active site', 'Binding site', 'Site', 'DNA binding', 'Biotechnology', 'GO annotation'],
    'Names and Taxonomy': ['Protein names', 'Gene names', 'Organism', 'Taxonomic lineage', 'Proteomes', 'Virus host'],
    'Disease and Variants': ['Involvement in disease', 'Natural variant', 'Allergenic properties', 'Toxic dose', 'Pharmaceutical use', 'Disruption phenotype', 'Mutagenesis'], 
    'Subcellular location': ['Subcellular location', 'Transmembrane', 'Topological domain', 'Intramembrane'], 
    'PTM/Processing': ['Signal peptide', 'Propeptide', 'Transit peptide', 'Chain', 'Peptide', 'Modified residue', 'Lipidation', 'Glycosylation', 'Disulfide bond', 'Cross-link', 'Post-translational modification'], 
    'Expression': ['Tissue specificity', 'Developmental stage', 'Induction'],
    'Interaction': ['Subunit'],
    'Family and Domains': ['Domain', 'Repeat', 'Compositional bias', 'Region', 'Coiled coil', 'Motif', 'Domain (non-positional annotation)', 'Sequence similarities'], 
    'Sequence': ['RNA Editing', 'Polymorphism']
}

subsection_rate = {'Sequence similarities': 0.909675970045262, 'Chain': 1.0107442527373975, 'Proteomes': 0.8600404006367225, 
                   'GO annotation': 5.555836944293805, 'Protein names': 2.590984796233018, 'Gene names': 1.967604024619467, 
                   'Organism': 1.5465265456051585, 'Taxonomic lineage': 1.0, 'Function': 0.8560389474774173, 
                   'Subcellular location': 0.8948881435889876, 'Post-translational modification': 0.11196873250461133, 
                   'Propeptide': 0.02604981107174009, 'Binding site': 2.0818209419912144, 'Modified residue': 0.455444345578131, 
                   'Lipidation': 0.024084184958397172, 'Glycosylation': 0.21670150387947903, 'Cofactor': 0.4048873889289619, 
                   'Domain': 0.3758522831975823, 'Catalytic activity': 0.5881363933919862, 'Subunit': 0.5187989322438149, 
                   'Activity regulation': 0.03193440424856044, 'Domain (non-positional annotation)': 0.10132978116614279, 
                   'Motif': 0.08284061053751099, 'Active site': 0.3094158755899072, 'Induction': 0.04422307750358463, 
                   'Topological domain': 0.2614984740072272, 'Transmembrane': 0.6684515253785146, 'Region': 0.5607861100434719, 
                   'Compositional bias': 0.3057812223035383, 'DNA binding': 0.021327043329770635, 'Site': 0.11362020944448246, 
                   'Coiled coil': 0.03935288780311447, 'Pathway': 0.25215121982895544, 'Repeat': 0.19109922375318755, 
                   'Miscellaneous': 0.07987988620428822, 'Caution': 0.02515474918084287, 'Signal peptide': 0.07696654750058354, 
                   'Natural variant': 0.11376061130972125, 'Disruption phenotype': 0.035189972498784645, 'Tissue specificity': 0.08883928022983785, 
                   'Disulfide bond': 0.2353328313966651, 'Developmental stage': 0.02477917419132913, 'Mutagenesis': 0.1590507429891206, 
                   'Virus host': 0.06191195750035539, 'Biophysicochemical properties': 0.013750607676822986, 'Cross-link': 0.04331397542616354, 
                   'Involvement in disease': 0.012013134594493088, 'Transit peptide': 0.014324500300986498, 'Peptide': 0.021871100557570907, 
                   'Biotechnology': 0.0031994075041286924, 'Intramembrane': 0.005210664223674211, 'RNA Editing': 0.005710845868587364, 
                   'Polymorphism': 0.0025991895302329092, 'Allergenic properties': 0.0016620070797640546, 'Toxic dose': 0.001493524841477519, 
                   'Pharmaceutical use': 0.00028957884705498315}
