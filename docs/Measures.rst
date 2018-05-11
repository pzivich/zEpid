.. image:: images/zepid_logo.png

-------------------------------------


Measures
'''''''''''''''''''''''''''''''''

For the following examples, we will load the following dataset....


.. code:: python
    df = pd.read_csv('datasets/MACS.csv')


Measures of Effect/Association
------------------------------

There are several association measures currently implemented. We can
calculate 

.. code:: python
    import zepid
    zepid.RiskRatio()

