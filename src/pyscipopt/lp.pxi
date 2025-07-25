##@file lp.pxi
#@brief Base class of the LP Plugin
cdef class LP:
    cdef SCIP_LPI* lpi
    cdef readonly str name

    def __init__(self, name="LP", sense="minimize"):
        """
        Keyword arguments:
        name -- the name of the problem (default 'LP')
        sense -- objective sense (default minimize)
        """
        self.name = name
        n = str_conversion(name)
        if sense == "minimize":
            PY_SCIP_CALL(SCIPlpiCreate(&(self.lpi), NULL, n, SCIP_OBJSEN_MINIMIZE))
        elif sense == "maximize":
            PY_SCIP_CALL(SCIPlpiCreate(&(self.lpi), NULL, n, SCIP_OBJSEN_MAXIMIZE))
        else:
            raise Warning("unrecognized objective sense")

    def __dealloc__(self):
        PY_SCIP_CALL(SCIPlpiFree(&(self.lpi)))

    def __repr__(self):
        return self.name

    def writeLP(self, filename):
        """Writes LP to a file.

        Keyword arguments:
        filename -- the name of the file to be used
        """
        PY_SCIP_CALL(SCIPlpiWriteLP(self.lpi, filename))

    def readLP(self, filename):
        """Reads LP from a file.

        Keyword arguments:
        filename -- the name of the file to be used
        """
        PY_SCIP_CALL(SCIPlpiReadLP(self.lpi, filename))

    def infinity(self):
        """Returns infinity value of the LP.
        """
        return SCIPlpiInfinity(self.lpi)

    def isInfinity(self, val):
        """Checks if a given value is equal to the infinity value of the LP.

        Keyword arguments:
        val -- value that should be checked
        """
        return SCIPlpiIsInfinity(self.lpi, val)

    def addCol(self, entries, obj = 0.0, lb = 0.0, ub = None):
        """Adds a single column to the LP.

        Keyword arguments:
        entries -- a list of tuples; if p is the index of the new column, then each tuple (i, k) indicates that
        A[i][p] = k, where A is the constraint matrix and k is a nonzero entry.
        obj     -- objective coefficient (default 0.0)
        lb      -- lower bound (default 0.0)
        ub      -- upper bound (default infinity)
        """
        cdef int nnonz = len(entries)
        cdef SCIP_Real* c_coefs  = <SCIP_Real*> malloc(nnonz * sizeof(SCIP_Real))
        cdef int* c_inds = <int*>malloc(nnonz * sizeof(int))
        cdef SCIP_Real c_obj = obj
        cdef SCIP_Real c_lb = lb
        cdef SCIP_Real c_ub = ub if ub != None else self.infinity()
        cdef int c_beg = 0
        cdef int i

        for i,entry in enumerate(entries):
            c_inds[i] = entry[0]
            c_coefs[i] = entry[1]

        PY_SCIP_CALL(SCIPlpiAddCols(self.lpi, 1, &c_obj, &c_lb, &c_ub, NULL, nnonz, &c_beg, c_inds, c_coefs))

        free(c_coefs)
        free(c_inds)

    def addCols(self, entrieslist, objs = None, lbs = None, ubs = None):
        """Adds multiple columns to the LP.

        Keyword arguments:
        entrieslist -- a list of lists, where the j-th inner list contains tuples (i, k) such that A[i][p] = k,
        where A is the constraint matrix, p is the index of the j-th new column, and k is a nonzero entry.
        objs  -- objective coefficient (default 0.0)
        lbs   -- lower bounds (default 0.0)
        ubs   -- upper bounds (default infinity)
        """
        cdef int ncols = len(entrieslist)
        cdef SCIP_Real* c_objs   = <SCIP_Real*> malloc(ncols * sizeof(SCIP_Real))
        cdef SCIP_Real* c_lbs    = <SCIP_Real*> malloc(ncols * sizeof(SCIP_Real))
        cdef SCIP_Real* c_ubs    = <SCIP_Real*> malloc(ncols * sizeof(SCIP_Real))
        cdef SCIP_Real* c_coefs
        cdef int* c_inds
        cdef int* c_beg
        cdef int nnonz = sum(len(entries) for entries in entrieslist)
        cdef int i

        if nnonz > 0:
            c_coefs  = <SCIP_Real*> malloc(nnonz * sizeof(SCIP_Real))
            c_inds = <int*>malloc(nnonz * sizeof(int))
            c_beg  = <int*>malloc(ncols * sizeof(int))

            tmp = 0
            for i,entries in enumerate(entrieslist):
                c_objs[i] = objs[i] if objs != None else 0.0
                c_lbs[i] = lbs[i] if lbs != None else 0.0
                c_ubs[i] = ubs[i] if ubs != None else self.infinity()
                c_beg[i] = tmp

                for entry in entries:
                    c_inds[tmp] = entry[0]
                    c_coefs[tmp] = entry[1]
                    tmp += 1

            PY_SCIP_CALL(SCIPlpiAddCols(self.lpi, ncols, c_objs, c_lbs, c_ubs, NULL, nnonz, c_beg, c_inds, c_coefs))

            free(c_beg)
            free(c_inds)
            free(c_coefs)
        else:
            for i in range(len(entrieslist)):
                c_objs[i] = objs[i] if objs != None else 0.0
                c_lbs[i] = lbs[i] if lbs != None else 0.0
                c_ubs[i] = ubs[i] if ubs != None else self.infinity()

            PY_SCIP_CALL(SCIPlpiAddCols(self.lpi, ncols, c_objs, c_lbs, c_ubs, NULL, 0, NULL, NULL, NULL))

        free(c_ubs)
        free(c_lbs)
        free(c_objs)

    def delCols(self, firstcol, lastcol):
        """Deletes a range of columns from the LP.

        Keyword arguments:
        firstcol -- first column to delete
        lastcol  -- last column to delete
        """
        PY_SCIP_CALL(SCIPlpiDelCols(self.lpi, firstcol, lastcol))

    def addRow(self, entries, lhs=0.0, rhs=None):
        """Adds a single row to the LP.

        Keyword arguments:
        entries -- a list of tuples; if q is the index of the new row, then each tuple (j, k) indicates that
        A[q][j] = k, where A is the constraint matrix and k is a nonzero entry.
        lhs     -- left-hand side of the row (default 0.0)
        rhs     -- right-hand side of the row (default infinity)
        """
        cdef int nnonz = len(entries)
        cdef SCIP_Real* c_coefs  = <SCIP_Real*> malloc(nnonz * sizeof(SCIP_Real))
        cdef int* c_inds = <int*>malloc(nnonz * sizeof(int))
        cdef SCIP_Real c_lhs = lhs
        cdef SCIP_Real c_rhs = rhs if rhs != None else self.infinity()
        cdef int c_beg = 0
        cdef int i

        for i,entry in enumerate(entries):
            c_inds[i] = entry[0]
            c_coefs[i] = entry[1]

        PY_SCIP_CALL(SCIPlpiAddRows(self.lpi, 1, &c_lhs, &c_rhs, NULL, nnonz, &c_beg, c_inds, c_coefs))

        free(c_coefs)
        free(c_inds)

    def addRows(self, entrieslist, lhss = None, rhss = None):
        """Adds multiple rows to the LP.

        Keyword arguments:
        entrieslist -- a list of lists, where the i-th inner list contains tuples (j, k) such that A[q][j] = k,
        where A is the constraint matrix, q is the index of the i-th new row, and k is a nonzero entry.
        lhss        -- left-hand side of the row (default 0.0)
        rhss        -- right-hand side of the row (default infinity)
        """
        cdef int nrows = len(entrieslist)
        cdef SCIP_Real* c_lhss  = <SCIP_Real*> malloc(nrows * sizeof(SCIP_Real))
        cdef SCIP_Real* c_rhss  = <SCIP_Real*> malloc(nrows * sizeof(SCIP_Real))
        cdef int* c_beg  = <int*>malloc(nrows * sizeof(int))
        cdef int nnonz = sum(len(entries) for entries in entrieslist)
        cdef SCIP_Real* c_coefs = <SCIP_Real*> malloc(nnonz * sizeof(SCIP_Real))
        cdef int* c_inds = <int*>malloc(nnonz * sizeof(int))
        cdef int tmp = 0
        cdef int i

        for i,entries in enumerate(entrieslist):
            c_lhss[i] = lhss[i] if lhss != None else 0.0
            c_rhss[i] = rhss[i] if rhss != None else self.infinity()
            c_beg[i]  = tmp

            for entry in entries:
                c_inds[tmp] = entry[0]
                c_coefs[tmp] = entry[1]
                tmp += 1

        PY_SCIP_CALL(SCIPlpiAddRows(self.lpi, nrows, c_lhss, c_rhss, NULL, nnonz, c_beg, c_inds, c_coefs))

        free(c_beg)
        free(c_inds)
        free(c_coefs)
        free(c_lhss)
        free(c_rhss)

    def delRows(self, firstrow, lastrow):
        """Deletes a range of rows from the LP.

        Keyword arguments:
        firstrow -- first row to delete
        lastrow  -- last row to delete
        """
        PY_SCIP_CALL(SCIPlpiDelRows(self.lpi, firstrow, lastrow))

    def getBounds(self, firstcol = 0, lastcol = None):
        """Returns all lower and upper bounds for a range of columns.

        Keyword arguments:
        firstcol -- first column (default 0)
        lastcol  -- last column (default ncols - 1)
        """
        cdef int i

        lastcol = lastcol if lastcol != None else self.ncols() - 1

        if firstcol > lastcol:
            return None

        ncols = lastcol - firstcol + 1
        cdef SCIP_Real* c_lbs = <SCIP_Real*> malloc(ncols * sizeof(SCIP_Real))
        cdef SCIP_Real* c_ubs = <SCIP_Real*> malloc(ncols * sizeof(SCIP_Real))
        PY_SCIP_CALL(SCIPlpiGetBounds(self.lpi, firstcol, lastcol, c_lbs, c_ubs))

        lbs = []
        ubs = []

        for i in range(ncols):
            lbs.append(c_lbs[i])
            ubs.append(c_ubs[i])

        free(c_ubs)
        free(c_lbs)

        return lbs, ubs

    def getSides(self, firstrow = 0, lastrow = None):
        """Returns all left- and right-hand sides for a range of rows.

        Keyword arguments:
        firstrow -- first row (default 0)
        lastrow  -- last row (default nrows - 1)
        """
        cdef int i

        lastrow = lastrow if lastrow != None else self.nrows() - 1

        if firstrow > lastrow:
            return None

        nrows = lastrow - firstrow + 1
        cdef SCIP_Real* c_lhss = <SCIP_Real*> malloc(nrows * sizeof(SCIP_Real))
        cdef SCIP_Real* c_rhss = <SCIP_Real*> malloc(nrows * sizeof(SCIP_Real))
        PY_SCIP_CALL(SCIPlpiGetSides(self.lpi, firstrow, lastrow, c_lhss, c_rhss))

        lhss = []
        rhss = []

        for i in range(firstrow, lastrow + 1):
            lhss.append(c_lhss[i])
            rhss.append(c_rhss[i])

        free(c_rhss)
        free(c_lhss)

        return lhss, rhss

    def chgObj(self, col, obj):
        """Changes objective coefficient of a single column.

        Keyword arguments:
        col -- column to change
        obj -- new objective coefficient
        """
        cdef SCIP_Real c_obj = obj
        cdef int c_col = col

        PY_SCIP_CALL(SCIPlpiChgObj(self.lpi, 1, &c_col, &c_obj))

    def chgCoef(self, row, col, newval):
        """Changes a single coefficient in the LP.

        Keyword arguments:
        row -- row to change
        col -- column to change
        newval -- new coefficient
        """
        PY_SCIP_CALL(SCIPlpiChgCoef(self.lpi, row, col, newval))

    def chgBound(self, col, lb, ub):
        """Changes the lower and upper bound of a single column.

        Keyword arguments:
        col -- column to change
        lb  -- new lower bound
        ub  -- new upper bound
        """
        cdef SCIP_Real c_lb = lb
        cdef SCIP_Real c_ub = ub
        cdef int c_col = col

        PY_SCIP_CALL(SCIPlpiChgBounds(self.lpi, 1, &c_col, &c_lb, &c_ub))

    def chgSide(self, row, lhs, rhs):
        """Changes the left- and right-hand side of a single row.

        Keyword arguments:
        row -- row to change
        lhs -- new left-hand side
        rhs -- new right-hand side
        """
        cdef SCIP_Real c_lhs = lhs
        cdef SCIP_Real c_rhs = rhs
        cdef int c_row = row

        PY_SCIP_CALL(SCIPlpiChgSides(self.lpi, 1, &c_row, &c_lhs, &c_rhs))

    def clear(self):
        """Clears the whole LP."""
        PY_SCIP_CALL(SCIPlpiClear(self.lpi))

    def nrows(self):
        """Returns the number of rows."""
        cdef int nrows

        PY_SCIP_CALL(SCIPlpiGetNRows(self.lpi, &nrows))

        return nrows

    def ncols(self):
        """Returns the number of columns."""
        cdef int ncols

        PY_SCIP_CALL(SCIPlpiGetNCols(self.lpi, &ncols))

        return ncols

    def solve(self, dual=True):
        """Solves the current LP.

        Keyword arguments:
        dual -- use the dual or primal Simplex method (default: dual)
        """
        if dual:
            PY_SCIP_CALL(SCIPlpiSolveDual(self.lpi))
        else:
            PY_SCIP_CALL(SCIPlpiSolvePrimal(self.lpi))

        cdef SCIP_Real objval
        PY_SCIP_CALL(SCIPlpiGetObjval(self.lpi, &objval))

        return objval

    def isOptimal(self):
        """
        returns true iff LP was solved to optimality.

        Returns
        -------
        bool

        """
        return SCIPlpiIsOptimal(self.lpi)

    def getObjVal(self):
        """
        Returns the objective value of the last LP solve.
        Please note that infeasible or unbounded LPs might return unexpected results.
        """
        cdef SCIP_Real objval

        PY_SCIP_CALL(SCIPlpiGetSol(self.lpi, &objval, NULL, NULL, NULL, NULL))

        return objval

    def getPrimal(self):
        """
        Returns the primal solution of the last LP solve.
        Please note that infeasible or unbounded LPs might return unexpected results.
        """
        cdef int ncols = self.ncols()
        cdef SCIP_Real* c_primalsol = <SCIP_Real*> malloc(ncols * sizeof(SCIP_Real))
        cdef int i

        PY_SCIP_CALL(SCIPlpiGetSol(self.lpi, NULL, c_primalsol, NULL, NULL, NULL))
        primalsol = [0.0] * ncols
        for i in range(ncols):
            primalsol[i] = c_primalsol[i]
        free(c_primalsol)

        return primalsol

    def isPrimalFeasible(self):
        """Returns True iff LP is proven to be primal feasible."""
        return SCIPlpiIsPrimalFeasible(self.lpi)

    def getDual(self):
        """
        Returns the dual solution of the last LP solve.
        Please note that infeasible or unbounded LPs might return unexpected results.
        """
        cdef int nrows = self.nrows()
        cdef SCIP_Real* c_dualsol = <SCIP_Real*> malloc(nrows * sizeof(SCIP_Real))
        cdef int i

        PY_SCIP_CALL(SCIPlpiGetSol(self.lpi, NULL, NULL, c_dualsol, NULL, NULL))
        dualsol = [0.0] * nrows
        for i in range(nrows):
            dualsol[i] = c_dualsol[i]
        free(c_dualsol)

        return dualsol

    def isDualFeasible(self):
        """Returns True iff LP is proven to be dual feasible."""
        return SCIPlpiIsDualFeasible(self.lpi)

    def getPrimalRay(self):
        """Returns a primal ray if possible, None otherwise."""
        cdef int ncols
        cdef SCIP_Real* c_ray
        cdef int i

        if not SCIPlpiHasPrimalRay(self.lpi):
            return None

        ncols = self.ncols()
        c_ray = <SCIP_Real*> malloc(ncols * sizeof(SCIP_Real))

        PY_SCIP_CALL(SCIPlpiGetPrimalRay(self.lpi, c_ray))
        ray = [0.0] * ncols
        for i in range(ncols):
            ray[i] = c_ray[i]
        free(c_ray)

        return ray

    def getDualRay(self):
        """Returns a dual ray if possible, None otherwise."""
        cdef int nrows
        cdef SCIP_Real* c_ray
        cdef int i

        if not SCIPlpiHasDualRay(self.lpi):
            return None

        nrows = self.nrows()
        c_ray = <SCIP_Real*> malloc(nrows * sizeof(SCIP_Real))

        PY_SCIP_CALL(SCIPlpiGetDualfarkas(self.lpi, c_ray))
        ray = [0.0] * nrows
        for i in range(nrows):
            ray[i] = c_ray[i]
        free(c_ray)

        return ray

    def getNIterations(self):
        """Returns the number of LP iterations of the last LP solve."""
        cdef int niters

        PY_SCIP_CALL(SCIPlpiGetIterations(self.lpi, &niters))

        return niters

    def getActivity(self):
        """
        Returns the row activity vector of the last LP solve.
        Please note that infeasible or unbounded LPs might return unexpected results.
        """
        cdef int nrows = self.nrows()
        cdef SCIP_Real* c_activity = <SCIP_Real*> malloc(nrows * sizeof(SCIP_Real))
        cdef int i

        PY_SCIP_CALL(SCIPlpiGetSol(self.lpi, NULL, NULL, NULL, c_activity, NULL))

        activity = [0.0] * nrows
        for i in range(nrows):
            activity[i] = c_activity[i]

        free(c_activity)

        return activity

    def getRedcost(self):
        """
        Returns the reduced cost vector of the last LP solve.
        Please note that infeasible or unbounded LPs might return unexpected results.
        """
        cdef int ncols = self.ncols()
        cdef SCIP_Real* c_redcost = <SCIP_Real*> malloc(ncols * sizeof(SCIP_Real))
        cdef int i

        PY_SCIP_CALL(SCIPlpiGetSol(self.lpi, NULL, NULL, NULL, NULL, c_redcost))

        redcost = [0.0] * ncols
        for i in range(ncols):
            redcost[i] = c_redcost[i]

        free(c_redcost)

        return redcost

    def getBasisInds(self):
        """Returns the indices of the basic columns and rows; index i >= 0 corresponds to column i, index i < 0 to row -i-1"""
        cdef int nrows = self.nrows()
        cdef int* c_binds = <int*> malloc(nrows * sizeof(int))
        cdef int i

        PY_SCIP_CALL(SCIPlpiGetBasisInd(self.lpi, c_binds))

        binds = []
        for i in range(nrows):
            binds.append(c_binds[i])

        free(c_binds)

        return binds

    # Parameter Methods

    def setIntParam(self, param, value):
        """
        Set an int-valued parameter.
        If the parameter is not supported by the LP solver, KeyError will be raised.

        Parameters
        ----------
        param : SCIP_LPPARAM
            name of parameter
        value : int
            value of parameter

        """
        PY_SCIP_CALL(SCIPlpiSetIntpar(self.lpi, param, value))

    def setRealParam(self, param, value):
        """
        Set a real-valued parameter.
        If the parameter is not supported by the LP solver, KeyError will be raised.

        Parameters
        ----------
        param : SCIP_LPPARAM
            name of parameter
        value : float
            value of parameter

        """
        PY_SCIP_CALL(SCIPlpiSetRealpar(self.lpi, param, value))

    def getIntParam(self, param):
        """
        Get the value of a parameter of type int.
        If the parameter is not supported by the LP solver, KeyError will be raised.

        Parameters
        ----------
        param : SCIP_LPPARAM
            name of parameter

        Returns
        -------
        int

        """
        cdef int value

        PY_SCIP_CALL(SCIPlpiGetIntpar(self.lpi, param, &value))

        return value

    def getRealParam(self, param):
        """
        Get the value of a parameter of type float.
        If the parameter is not supported by the LP solver, KeyError will be raised.

        Parameters
        ----------
        param : SCIP_LPPARAM
            name of parameter

        Returns
        -------
        float

        """
        cdef SCIP_Real value

        PY_SCIP_CALL(SCIPlpiGetRealpar(self.lpi, param, &value))

        return value
