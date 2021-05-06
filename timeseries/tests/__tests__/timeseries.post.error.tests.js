const rest = require('rest');
const mime = require('rest/interceptor/mime');

const timeseriesServiceBase = 'http://localhost:8001/timeseries-service/api/v1';

const callRESTService =  rest.wrap(mime, { mime: 'application/json' } );

describe("When a values POST request is missing the datasetId property", async () => {
    
	var response;
	
	beforeAll(async () => {
		response = await callRESTService({
		    method: 'POST',
		    path: timeseriesServiceBase + '/timeseries',
		    entity: {
		    	variableName: 'uint16_variable',
		    	boundaryGeometry: {
		    		type: 'Point',
		    		coordinates: [-123, 45]
		    	},
	    		start: 1,
	    		end: 5
		    }
		});
    });

    it ('HTTP response status code should be 422', async function() {
        expect(response.status.code).toBe(422);
    });

    it ('Error detail should show missing property', async function() {
        expect(response.entity.detail[0].loc).toEqual(['body', 'datasetId']);
    });
});

describe("When a values POST request is missing the variableName property", async () => {
    
	var response;
	
	beforeAll(async () => {
		response = await callRESTService({
		    method: 'POST',
		    path: timeseriesServiceBase + '/timeseries',
		    entity: {
		    	datasetId: 'annual_5x5x5_dataset',
		    	boundaryGeometry: {
		    		type: 'Point',
		    		coordinates: [-123, 45]
		    	},
	    		start: 1,
	    		end: 5
		    }
		});
    });

    it ('HTTP response status code should be 422', async function() {
        expect(response.status.code).toBe(422);
    });

    it ('Error detail should show missing property', async function() {
        expect(response.entity.detail[0].loc).toEqual(['body', 'variableName']);
    });    
});

describe("When a values POST request is missing the boundaryGeometry property", async () => {
    
	var response;
	
	beforeAll(async () => {
		response = await callRESTService({
		    method: 'POST',
		    path: timeseriesServiceBase + '/timeseries',
		    entity: {
		    	datasetId: 'annual_5x5x5_dataset',
		    	variableName: 'uint16_variable',
	    		start: 1,
	    		end: 5
		    }
		});
    });

    it ('HTTP response status code should be 422', async function() {
        expect(response.status.code).toBe(422);
    });

    it ('Error detail should show missing property', async function() {
        expect(response.entity.detail[0].loc).toEqual(['body', 'boundaryGeometry']);
    });
});

describe("When a values POST request specifies an unsupported boundary geometry type", async () => {
    
	var response;
	
	beforeAll(async () => {
		response = await callRESTService({
		    method: 'POST',
		    path: timeseriesServiceBase + '/timeseries',
		    entity: {
		    	datasetId: 'annual_5x5x5_dataset',
		    	variableName: 'uint16_variable',
		    	boundaryGeometry: {
		    		type: 'NoSuchGeometry',
		    		coordinates: [-123, 45]
		    	},		    	
	    		start: 1,
	    		end: 5
		    }
		});
    });


    it ('HTTP response status code should be 422', async function() {
        expect(response.status.code).toBe(422);
    });

    it ('Error detail should show missing property', async function() {
        expect(response.entity.detail[0].loc).toEqual(['body', 'boundaryGeometry', 'type']);
    });
});

describe("When a values POST request specifies coordinates outside of raster file", async () => {
    
	var response;
	
	beforeAll(async () => {
		response = await callRESTService({
		    method: 'POST',
		    path: timeseriesServiceBase + '/timeseries',
		    entity: {
		    	datasetId: 'annual_5x5x5_dataset',
		    	variableName: 'uint16_variable',
		    	boundaryGeometry: {
		    		type: 'Point',
		    		coordinates: [-124, 45]
		    	},
	    		start: 1,
	    		end: 5
		    }
		});
    });

    it ('HTTP response status code should be 422', async function() {
        expect(response.status.code).toBe(422);
    });

    it ('Error detail should show missing property', async function() {
        expect(response.entity.detail[0].type).toBe('value_error.selectedareaoutofbounds');
    });
});

describe("When a values POST request specifies a dataset that does not exist", async () => {
    
	var response;
	
	beforeAll(async () => {
		response = await callRESTService({
		    method: 'POST',
		    path: timeseriesServiceBase + '/timeseries',
		    entity: {
		    	datasetId: 'not-a-dataset',
		    	variableName: 'uint16_variable',
		    	boundaryGeometry: {
		    		type: 'Point',
		    		coordinates: [-123, 45]
		    	},
	    		start: 1,
	    		end: 5
		    }
		});
    });

    it ('HTTP response status code should be 422 - bad request', async function() {
        expect(response.status.code).toBe(422);
    });
    
    it ('Error summary should be bad request', async function() {
        expect(response.entity.detail[0].type).toBe("value_error.datasetnotfound");
    });
    
});

describe("When a values POST request specifies a nonexistent variable for dataset", async () => {
    
	var response;
	
	beforeAll(async () => {
		response = await callRESTService({
		    method: 'POST',
		    path: timeseriesServiceBase + '/timeseries',
		    entity: {
		    	datasetId: 'annual_5x5x5_dataset',
		    	variableName: 'not-a-variable',
		    	boundaryGeometry: {
		    		type: 'Point',
		    		coordinates: [-123, 45]
		    	},
	    		start: 1,
	    		end: 5
		    }
		});
    });

    it ('HTTP response status code should be 422', async function() {
        expect(response.status.code).toBe(422);
    });
    
    it ('Error summary should be variable not found', async function() {
        expect(response.entity.detail[0].type).toBe("value_error.variablenotfound");
    });
})

describe("When a values POST request specifies a range start outside of dataset coverage", async () => {
    
	var response;
	
	beforeAll(async () => {
		response = await callRESTService({
		    method: 'POST',
		    path: timeseriesServiceBase + '/timeseries',
		    entity: {
		    	datasetId: 'annual_5x5x5_dataset',
		    	variableName: 'uint16_variable',
		    	boundaryGeometry: {
		    		type: 'Point',
		    		coordinates: [-123, 45]
		    	},
	    		start: 6,
	    		end: 6
		    }
		});
    });

    it ('HTTP response status code should be 422', async function() {
        expect(response.status.code).toBe(422);
    });
    
    it ('Error summary should be selected area out of bounds', async function() {
        expect(response.entity.detail[0].type).toBe("value_error.timerangecontainment");
    });
})

describe("When a values POST request specifies a range end outside of dataset coverage", async () => {
    
	var response;
	
	beforeAll(async () => {
		response = await callRESTService({
		    method: 'POST',
		    path: timeseriesServiceBase + '/timeseries',
		    entity: {
		    	datasetId: 'annual_5x5x5_dataset',
		    	variableName: 'uint16_variable',
		    	boundaryGeometry: {
		    		type: 'Point',
		    		coordinates: [-123, 46]
		    	},
	    		start: 4,
	    		end: 6
		    }
		});
    });

    it ('HTTP response status code should be 422 - success', async function() {
        expect(response.status.code).toBe(422);
    });


    it ('Error summary should be validation error', async function() {
        expect(response.entity.detail[0].type).toBe("value_error.timerangecontainment");
    });
})


describe("When a values POST request specifies a range end before range start", async () => {
    
	var response;
	
	beforeAll(async () => {
		response = await callRESTService({
		    method: 'POST',
		    path: timeseriesServiceBase + '/timeseries',
		    entity: {
		    	datasetId: 'annual_5x5x5_dataset',
		    	variableName: 'uint16_variable',
		    	boundaryGeometry: {
		    		type: 'Point',
		    		coordinates: [-123, 45]
		    	},
	    		start: 4,
	    		end: 3
		    }
		});
    });

    it ('HTTP response status code should be 422', async function() {
        expect(response.status.code).toBe(422);
    });
    
    it ('Error summary should be validation error', async function() {
        expect(response.entity.detail[0].type).toBe("value_error.timerangeinvalid");
    });
})