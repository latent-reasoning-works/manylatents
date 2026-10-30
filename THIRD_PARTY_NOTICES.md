# Third-party notices

manylatents is licensed under Apache-2.0 (see `LICENSE` and `NOTICE`). Nine of its files are
ported from KrishnaswamyLab repositories whose code Yale University publishes under a
non-commercial license. Each of these files starts with the line
`# SPDX-License-Identifier: LicenseRef-Yale-Non-Commercial`. This document lists each file
under its upstream repository, with the upstream license text copied verbatim.

The Apache-2.0 license in LICENSE does not apply to the files listed in this document.

The last section, `archetypes`, is a separate BSD-3-Clause notice for code adapted in an
Apache-2.0 file. It lists no Yale-derived file.

## KrishnaswamyLab/MIOFlow

- Upstream: https://github.com/KrishnaswamyLab/MIOFlow
- License: Yale Non-Commercial License, Copyright 2024 Yale University
  (SPDX identifier `LicenseRef-Yale-Non-Commercial`)
- License text copied from `LICENSE.md` at commit `4e79ada3ce96feeb3f51ba02c7d999e095eaf7e8`

`gaga_net.py` follows `train_gaga_two_phase` of the `mioflow` 2.0 package. The two config
files configure the MIOFlow module and its network.

### Derived files

- `manylatents/algorithms/lightning/mioflow.py`
- `manylatents/algorithms/lightning/networks/mioflow_net.py`
- `manylatents/algorithms/lightning/networks/gaga_net.py`
- `manylatents/configs/algorithms/lightning/mioflow.yaml`
- `manylatents/configs/algorithms/lightning/mioflow_gaga.yaml`

### License text

```text
Non-Commercial License
Yale Copyright © 2024 Yale University. 
 
Permission is hereby granted to use, copy, modify, and distribute this Software for any non-commercial purpose. Any distribution or modification or derivations of the Software (together “Derivative Works”) must be made available on GitHub and shall include this copyright notice and this permission notice in all copies or substantial portions of the Software.  For the purposes of this license, "non-commercial" means not intended for or directed towards commercial advantage or monetary compensation either via the Software itself or Derivative Works or uses of either which lead to or generate any commercial products. In any event, the use and modification of the Software or Derivative Works shall remain governed by the terms and conditions of this Agreement; Any commercial use of the Software requires a separate commercial license from the copyright holder at Yale University. Direct any requests for commercial licenses to Yale Ventures at yaleventures@yale.edu.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 
```

## KrishnaswamyLab/cflows

- Upstream: https://github.com/KrishnaswamyLab/cflows
- License: Yale Non-Commercial License, Copyright 2024 Yale University
  (SPDX identifier `LicenseRef-Yale-Non-Commercial`)
- License text copied from `LICENSE.md` at commit `3878169cfa15582d0ebb54abacf1ea1e293e3912`

### Derived files

- `manylatents/algorithms/cflows_granger.py`
- `manylatents/algorithms/lightning/cflows.py`
- `manylatents/algorithms/lightning/losses/cflows.py`

### License text

```text
Non-Commercial License
Yale Copyright © 2024 Yale University. 
 
Permission is hereby granted to use, copy, modify, and distribute this Software for any non-commercial purpose. Any distribution or modification or derivations of the Software (together “Derivative Works”) must be made available on GitHub and shall include this copyright notice and this permission notice in all copies or substantial portions of the Software.  For the purposes of this license, "non-commercial" means not intended for or directed towards commercial advantage or monetary compensation either via the Software itself or Derivative Works or uses of either which lead to or generate any commercial products. In any event, the use and modification of the Software or Derivative Works shall remain governed by the terms and conditions of this Agreement; Any commercial use of the Software requires a separate commercial license from the copyright holder at Yale University. Direct any requests for commercial licenses to Yale Ventures at yaleventures@yale.edu.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 
```

## KrishnaswamyLab/AAnet

- Upstream: https://github.com/KrishnaswamyLab/AAnet
- License: Yale Non-Commercial License, Copyright 2024 Yale University
  (SPDX identifier `LicenseRef-Yale-Non-Commercial`)
- License text copied from `LICENSE.md` at commit `6dc3814419572d3295bdd32c8c62b8e6a81e2bbb`

### Derived files

- `manylatents/algorithms/lightning/networks/aanet.py`

### License text

```text
----------------------------------

Non-Commercial License
Yale Copyright © 2024 Yale University. 
 
Permission is hereby granted to use, copy, modify, and distribute this Software for any non-commercial purpose. Any distribution or modification or derivations of the Software (together “Derivative Works”) must be made available on GitHub and shall include this copyright notice and this permission notice in all copies or substantial portions of the Software.  For the purposes of this license, "non-commercial" means not intended for or directed towards commercial advantage or monetary compensation either via the Software itself or Derivative Works or uses of either which lead to or generate any commercial products. In any event, the use and modification of the Software or Derivative Works shall remain governed by the terms and conditions of this Agreement; Any commercial use of the Software requires a separate commercial license from the copyright holder at Yale University. Direct any requests for commercial licenses to Yale Ventures at yaleventures@yale.edu.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 

----------------------------------
```

## archetypes

- Upstream: https://github.com/aleixalcacer/archetypes
- License: BSD-3-Clause, Copyright (c) 2021, Aleix Alcacer Sales
- License text copied from `LICENSE` at commit `4518807340135213dbcb94b3512621b28c6eca41`

`manylatents/algorithms/latent/aa.py` holds two functions, `_optimize_alphas` and
`_aa_simple`, adapted from `archetypes` 0.6.2. `aa.py` is not a Yale-derived file.

`aa.py` is licensed under Apache-2.0; the BSD-3-Clause terms below cover only the two functions adapted from archetypes.

### License text

```text
BSD 3-Clause License

Copyright (c) 2021, Aleix Alcacer Sales
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
