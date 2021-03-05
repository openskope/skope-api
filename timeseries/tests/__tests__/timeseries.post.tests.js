const rest = require('rest');
const mime = require('rest/interceptor/mime');

const timeseriesServiceBase = 'http://localhost:8001/timeseries-service/api/v1';

const callRESTService =  rest.wrap(mime, { mime: 'application/json' } );

describe("When a values POST request selects first pixel of each band in 5x5x5 data cube", async () => {
    
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
	    		start: 0,
	    		end: 4
		    }
		});
    });

    it ('HTTP response status code should be 200 - success', async function() {
        expect(response.status.code).toBe(200);
    });
    
    it ('Dataset id should match that in the request', async function() {
        expect(response.entity.datasetId).toBe('annual_5x5x5_dataset');
    });

	it ('Variable name should match that in the request', async function() {
        expect(response.entity.variableName).toBe('uint16_variable');
    });

	it ('Boundary geometry type should be point', async function() {
        expect(response.entity.boundaryGeometry.type).toBe('Point');
    });

    it ('Boundary geometry coordinates should be array with requested lat and lng ', async function() {
        expect(response.entity.boundaryGeometry.coordinates).toEqual( [-123, 45] );
    });

    it ('Series range start and end should match the request', async function() {
        expect(response.entity.start).toEqual( "0" );
        expect(response.entity.end).toEqual( "4" );
    });

    it ('Values should be an array with one element for the first pixel of each band', async function() {
        expect(response.entity.values).toEqual( [100,200,300,400,500] );
    });
});

describe("When a values POST request selects first pixel of middle three bands in 5x5x5 data cube using string values for start and end", async () => {
    
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
	    		start: '1', 
	    		end: '3' 
		    }
		});
    });

    it ('HTTP response status code should be 200 - success', async function() {
        expect(response.status.code).toBe(200);
    });
    
    it ('Dataset id should match that in the request', async function() {
        expect(response.entity.datasetId).toBe('annual_5x5x5_dataset');
    });

	it ('Variable name should match that in the request', async function() {
        expect(response.entity.variableName).toBe('uint16_variable');
    });

	it ('Boundary geometry type should be point', async function() {
        expect(response.entity.boundaryGeometry.type).toBe('Point');
    });

    it ('Boundary geometry coordinates should be array with requested lat and lng ', async function() {
        expect(response.entity.boundaryGeometry.coordinates).toEqual( [-123, 45] );
    });

    it ('Series range start and end should match the request', async function() {
        expect(response.entity.start).toEqual( "1" );
        expect(response.entity.end).toEqual( "3" );
    });

    it ('Values should be an array with one element for the first pixel of each band', async function() {
        expect(response.entity.values).toEqual( [200,300,400] );
    });
});

describe("When a values POST request selects first pixel of first band in 5x5x5 data cube", async () => {
    
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
	    		start: 0,
	    		end: 0
		    }
		});
    });

    it ('HTTP response status code should be 200 - success', async function() {
        expect(response.status.code).toBe(200);
    });

    it ('Series range start and end should match the request', async function() {
        expect(response.entity.start).toEqual( "0" );
        expect(response.entity.end).toEqual( "0" );
    });
    
    it ('Values should be an array with one element for the first pixel of each band', async function() {
        expect(response.entity.values).toEqual( [100] );
    });
});

describe("When a values POST request selects last pixel of each band in 5x5x5 data cube", async () => {
    
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
		    		coordinates: [-119, 41]
		    	},
	    		start: 0,
	    		end: 4
		    }
		});
    });

    it ('HTTP response status code should be 200 - success', async function() {
        expect(response.status.code).toBe(200);
    });

    it ('Series range start and end should match the request', async function() {
        expect(response.entity.start).toEqual( "0" );
        expect(response.entity.end).toEqual( "4" );
    });

    it ('Values should be an array with one element for the first pixel of each band', async function() {
        expect(response.entity.values).toEqual( [144,244,344,444,544] );
    });

});


describe("When a values POST request selects last pixel of last band in 5x5x5 data cube", async () => {
    
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
		    		coordinates: [-119, 41]
		    	},
	    		start: 4,
	    		end: 4
		    }
		});
	});

    it ('HTTP response status code should be 200 - success', async function() {
        expect(response.status.code).toBe(200);
    });

    it ('Series range start and end should match the request', async function() {
        expect(response.entity.start).toEqual( "4" );
        expect(response.entity.end).toEqual( "4" );
    });

    it ('Values should be an array with one element for the first pixel of each band', async function() {
        expect(response.entity.values).toEqual( [544] );
    });

});

describe("When a values POST request selects last pixel of 5x5x5 data cube without specifying the bands", async () => {
    
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
		    		coordinates: [-119, 41]
		    	}
		    }
		});
	});

    it ('HTTP response status code should be 200 - success', async function() {
        expect(response.status.code).toBe(200);
    });

    it ('Series range start and end should represent all of the bands in the file', async function() {
        expect(response.entity.start).toEqual( "0" );
        expect(response.entity.end).toEqual( "4" );
    });

    it ('Values should be an array with one element for the first pixel of each band', async function() {
        expect(response.entity.values).toEqual( [144,244,344,444,544] );
    });

})



describe("When a values POST request selects last pixel of 5x5x5 data cube and specifies start but not end band", async () => {
    
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
		    		coordinates: [-119, 41]
		    	},
	    		start: 2
		    }
		});
	});

    it ('HTTP response status code should be 200 - success', async function() {
        expect(response.status.code).toBe(200);
    });

    it ('Series range start and end should represent bands from the requested start through the last band', async function() {
        expect(response.entity.start).toEqual( "2" );
        expect(response.entity.end).toEqual( "4" );
    });

    it ('Values should be an array with one element for the first requested band through the last band', async function() {
        expect(response.entity.values).toEqual( [344,444,544] );
    });

});


describe("When a values POST request selects last pixel of 5x5x5 data cube and specifies end but not start band", async () => {
    
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
		    		coordinates: [-119, 41]
		    	},
	    		end: 3
		    }
		});
	});

    it ('HTTP response status code should be 200 - success', async function() {
        expect(response.status.code).toBe(200);
    });

    it ('Series range start and end should represent bands from the first band through the requested end band', async function() {
        expect(response.entity.start).toEqual( "0" );
        expect(response.entity.end).toEqual( "3" );
    });

    it ('Values should be an array with one element for the first band through the requested end band', async function() {
        expect(response.entity.values).toEqual( [144,244,344,444] );
    });

});
