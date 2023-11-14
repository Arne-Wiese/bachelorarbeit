# Bachelorarbeit 
## The missing link: On hierarchies from multiple subspaces to single subspace optimized clustering
In the setting of high-dimensional clustering, a sub-taxonomy exists differentiating between clusters that reside in a single common arbitrarily oriented subspace (SSO clustering) and clusters that reside in multiple arbitrarily oriented subspaces (MSO clustering). While there exists in the literature a plethora of methods for either archetype, they are regarded in a binary fashion as “its either SSO or MSO clustering” in two disjoint categories. We challenge this circumstance by investigating transitions between both worlds in a hierarchical fashion. In this thesis, we will together discover the properties of bridging both worlds (MSO and SSO). The tasks are to implement and evaluate a prototype of this concept.


# PCAfold

## Installation

### Dependencies

**PCAfold** requires Python>=3.8 and the latest versions of the packages in requirements.txt.
Change to root of this repo and install all necessary dependencies:

```bash
conda install --file requirements.txt
```

### Build from source

Clone the `PCAfold` repository and move into the `PCAfold` directory created:

```bash
git clone http://gitlab.multiscale.utah.edu/common/PCAfold.git
cd PCAfold
```

Run the installation:

```bash
python -m pip install .
```


