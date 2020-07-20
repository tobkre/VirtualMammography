# Virtual Mammography
This software packeage implements a method for the simulation of realistic mammography images of virtual objects. 
Currently, single cells of the contrast-detail phantom for mammography (CDMAM) are implemented along with a virtual 
representation of the whole phantom. Details of the implementation are described in *Determination of contrast-detail curves in mammography image quality assessment by a parametric model observer* [1].

## Requirements

You need to have ```python3``` installed in order to run this software.

Furthermore the following packages are required:
1. ```pydicom (2.0.0)```
2. ```yaml (5.1.2)```
3. ```h5py (2.9.0)```


## Usage

You can start the software using the command line by choosing the desired configuration file that is located in .\config\ using the following command:

```python3 cdmam_simulator.py --config config-file.yml```

This simulates images according to the chosen configuration and saves DICOM images in the .\Results\DICOM\ folder.

## Citation

```@article{kretz2019determination,
  title={Determination of contrast-detail curves in mammography image quality assessment by a parametric model observer},
  author={Kretz, T and Anton, M and Schaeffter, T and Elster, C},
  journal={Physica Medica},
  volume={62},
  pages={120--128},
  year={2019},
  publisher={Elsevier}
}
```
## References

[1] Kretz, T., Anton, M., Schaeffter, T. and Elster, C. (2019). Determination of contrast-detail curves in mammography image quality assessment by a parametric model observer. Physica Medica, 62, 120-128.

## License

copyright: Tobias Kretz(PTB), 2020.

This software is licensed under the BSD-like license:

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 
1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in
   the documentation and/or other materials provided with the distribution.
   
## DISCLAIMER
 
This software was developed at Physikalisch-Technische Bundesanstalt
(PTB). The software is made available "as is" free of cost. PTB assumes
no responsibility whatsoever for its use by other parties, and makes no
guarantees, expressed or implied, about its quality, reliability, safety,
suitability or any other characteristic. In no event will PTB be liable
for any direct, indirect or consequential damage arising in connection
with the usage of this software.
