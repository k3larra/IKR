'use strict';
//const math = require('mathjs');

let study_nbr=0;
const version = "version07";
//const models = ["ResNet101","ResNet152","GoogLeNet","Inception_V3","efficientnet_v2_s"]; //remove
//const lowest_mean_class_probability = 16; //Remove
const number_of_probabilities_to_show = 10;  //Remove
let imagenet_classes =[];
let probs =[]

//Move this to another file......

const params = new Proxy(new URLSearchParams(window.location.search), {
  get: (searchParams, prop) => searchParams.get(prop),
});
// Get the value of "some_key" in eg "https://example.com/?some_key=some_value"
 // "some_value"


const Prob = function(probability,classnumber,label){
  this.probability = probability;
  this.classnumber = classnumber;
  this.label = label;
}

let mean_and_dispersion;
class Mean_and_Dispersion{
    constructor(image,value){
      this.image=image;
      this.value=value;
    }
    get_mean_values(classnumber){
      if (this.value.hasOwnProperty("mean_average_csv_for_candidate_"+classnumber)){
        let result= []
        let itBe = (this.value["mean_average_csv_for_candidate_"+classnumber]).split(',');
        for (let i =0;i<itBe.length;i++){
          result.push(itBe[i])
        }
        return result;
      }else{
        return"";
      }

    }
    get_std_values(classnumber){
      if (this.value.hasOwnProperty("std_average_csv_for_candidate_"+classnumber)){
        let result= []
        let itBe = (this.value["std_average_csv_for_candidate_"+classnumber]).split(',');
        for (let i =0;i<itBe.length;i++){
          result.push(itBe[i])
        }
        return result;
      }else{
        return"";
      }
    }
    get_mean_saliency_map_path(classnumber){
      if (this.value.hasOwnProperty("mean_image_path_candidate_"+classnumber)){
        return "../"+this.value["mean_image_path_candidate_"+classnumber];
      }else{
        return"";
      }
    }
    get_dispersion_saliency_map_path(classnumber){
      if (this.value.hasOwnProperty("diff_image_path_candidate_"+classnumber)){
        return "../"+this.value["diff_image_path_candidate_"+classnumber];
      }else{
        return"";
      }
    }
}

let probs_v1 =[]
class Prob_v1{
  constructor(classnumber,label){
    this.classnumber = classnumber;
    this.label = label;
    this.probs =[];
    this.maxProbability=0;
  }
  get_average_prob(){
    var average =0;
    this.probs.forEach((item,index,array) => {
      average=average+item;
    });
    return average/this.probs.length;
  }
  get_number_prob(){
    return this.probs.length;
  }
}

let models_v1 = []
class Model_v1{
  constructor(name,values){
    this.name = name
    this.values = values;
  }
  get_saliency_map_for_class(classnumber){
    let path="";
      Object.entries(this.values.xai).forEach(([key, value]) => {
        if (Number.isInteger(parseInt(key))){
          //let path_label = value.substring(value.lastIndexOf("/")+1).split(".")[0].replace("_"," ");
          let path_label = value.substring(value.lastIndexOf("/")+1).split(".")[0];
          if(parseInt(path_label)===classnumber){
            path=value;
          }
        }
      });
    return path;
  }
  get_model_name(){
    return this.name;
  }
  get_class_prob_for_label(classnumber){
    let probability=0;
      Object.entries(this.values).forEach(([key, value]) => {
        if (Number.isInteger(parseInt(key))){
          if(classnumber===value.labelid){
            probability=value.probability;
          }
        }
      });
    return probability;
  }
  get_saliency_values(classnumber){
    let mean_values=[]
      Object.entries(this.values.xai).forEach(([key, value]) => {
        if(value.hasOwnProperty("target_idx")) {
          if (parseInt(value.target_idx)===classnumber){
            let mean_string_values = (value.mean_values).split(",");
            for (var j=0;j<mean_string_values.length;++j){
                 mean_values.push(mean_string_values[j]);
             }
          }
        }
      });
    return  mean_values;
  }
  get_raw_values(classnumber){
    let raw_values=[]
      Object.entries(this.values.xai).forEach(([key, value]) => {
        if(value.hasOwnProperty("target_idx")) {
          if (parseInt(value.target_idx)===classnumber){
            raw_values = csv_to_list(value.raw_string)
          }
        }
      });
    return  raw_values;
  }

  get_mean_raw_value(classnumber){
    let mean_value=0;
      Object.entries(this.values.xai).forEach(([key, value]) => {
        if(value.hasOwnProperty("target_idx")) {
          if (parseInt(value.target_idx)===classnumber){
            mean_value = math.mean(csv_to_list(value.raw_string));
          }
        }
      });
   return  mean_value;
   }

   get_mean_from_pos_value(classnumber){
     let mean_value=0;
       Object.entries(this.values.xai).forEach(([key, value]) => {
         if(value.hasOwnProperty("target_idx")) {
           if (parseInt(value.target_idx)===classnumber){
             mean_value = math.mean(remove_neg_from_list(csv_to_list(value.raw_string)));
           }
         }
       });
    return  mean_value;
    }

    get_mean_from_neg_value(classnumber){
      let mean_value=0;
        Object.entries(this.values.xai).forEach(([key, value]) => {
          if(value.hasOwnProperty("target_idx")) {
            if (parseInt(value.target_idx)===classnumber){
              mean_value = math.mean(remove_pos_from_list(csv_to_list(value.raw_string)));
            }
          }
        });
     return  mean_value;
     }

     get_std_from_pos_value(classnumber){
       let std_value=0;
         Object.entries(this.values.xai).forEach(([key, value]) => {
           if(value.hasOwnProperty("target_idx")) {
             if (parseInt(value.target_idx)===classnumber){
               std_value = math.std(remove_neg_from_list(csv_to_list(value.raw_string)));
             }
           }
         });
      return  std_value;
      }
      get_std_from_neg_value(classnumber){
        let std_value=0;
          Object.entries(this.values.xai).forEach(([key, value]) => {
            if(value.hasOwnProperty("target_idx")) {
              if (parseInt(value.target_idx)===classnumber){
                std_value = math.std(remove_pos_from_list(csv_to_list(value.raw_string)));
              }
            }
          });
       return  std_value;
       }

   get_std_raw_value(classnumber){
     let std_value=0;
       Object.entries(this.values.xai).forEach(([key, value]) => {
         if(value.hasOwnProperty("target_idx")) {
           if (parseInt(value.target_idx)===classnumber){
             std_value = math.std(csv_to_list(value.raw_string));
           }
         }
       });
    return  std_value;
    }
}
function idx_to_label(index){
  return imagenet_classes[index];
}
function label_to_idx(label){
  return imagenet_classes.findIndex(x => x === label);
}

function csv_to_list(csv_string){
  let result= [];
    let itBe = (csv_string).split(',');
    for (let i =0;i<itBe.length;i++){
      result.push(itBe[i])
    }
    return result;
}

function remove_neg_from_list(list){
  let result= [];
    for (let i =0;i<list.length;i++){
      if (list[i]>0){
        result.push(list[i])
      }
    }
    return result;
}

function remove_pos_from_list(list){
  let result= [];
    for (let i =0;i<list.length;i++){
      if (list[i]<0){
        result.push(list[i])
      }
    }
    return result;
}
///until here
///Lets go
if (document.readyState != 'loading') {
  onDocumentReady();
} else {
  document.addEventListener('DOMContentLoaded', onDocumentReady);
}
// Page is loaded! Now event can be wired-up
function onDocumentReady() {
  console.log('Document ready.');
  if (params.study_nbr != null){
    study_nbr = params.study_nbr;
  }
  populate();
}
async function populate() {
  let requestURL = version+'/structure.json';
  let request = new Request(requestURL);
  let response = await fetch(request);
  const structure = await response.json();
  set_globals(structure[study_nbr]);
  requestURL = 'imagenet_classes.txt';
  request = new Request(requestURL);
  response = await fetch(request);
  const imagenet_classes_csv = await response.text();
  imagenet_classes = imagenet_classes_csv.split('\n');
  set_header_images(structure);
  set_predictions();
  compare_models();
  //populate_structure(structure);
}
function set_header_images(structure){
  var e = document.getElementById('org_image');
  var img = document.createElement("img");
  img.src="../"+structure[study_nbr]["image_path"];
  var cap = document.createElement("figcaption");
  cap.textContent ="Width:"+img.width+" Height:"+img.height
  cap.classList.add("text-center")
  img.classList.add("img-thumbnail");
  var fig = document.createElement("figure");
  fig.appendChild(img)
  fig.appendChild(cap)
  e.appendChild(fig)
  e = document.getElementById('trans_image');
  img = document.createElement("img");
  img.src="../"+structure[study_nbr]["image_path_transformed"];
  cap = document.createElement("figcaption");
  cap.textContent ="Width:"+img.width+" Height:"+img.height //ToDo: Why is reload needed
  cap.classList.add("text-center")
  img.classList.add("img-thumbnail");
  fig = document.createElement("figure");
  fig.appendChild(img)
  fig.appendChild(cap)
  e.appendChild(fig)
}
function set_globals(study){

  Object.entries(study).forEach(([key, value]) => {
    if (value.hasOwnProperty('xai')){
      models_v1.push(new Model_v1(key,value))
    }
  });
  models_v1.forEach((model, i) => {
    for (let j = 0; j < number_of_probabilities_to_show; j++) {
      var t = probs_v1.find(element => element.classnumber === model.values[j].labelid);
      if (t === undefined) {
          const prob = new Prob_v1(model.values[j].labelid, model.values[j].label);
          prob.probs.push(model.values[j].probability);
          prob.maxProbability=model.values[j].probability;
          const prob_ref = probs_v1.push(prob);
      }else{
        if (t.maxProbability<model.values[j].probability){
          t.maxProbability=model.values[j].probability;
        }
        t.probs.push(model.values[j].probability)
      }
    }
    probs_v1.sort((c1, c2) => (c1.maxProbability < c2.maxProbability) ? 1 : (c1.maxProbability > c2.maxProbability) ? -1 : 0);
  });
   mean_and_dispersion = new Mean_and_Dispersion(study_nbr,study['diff_mean_maps']);
}
function set_predictions(){
  const e = document.getElementById('predictions');
  let row_model,column_model,model_name;
  let labels,preds,label,pred;
  let row_models=[];
  let columns_pred_labels=[];
  let columns_pred_values=[];
  let row_pred;
  let column,row;
  models_v1.forEach((model, i) => {
    column = i%4;
    row = Math.trunc((i/4));
      if (column===0){
      row_models = [];
      columns_pred_labels = [];
      columns_pred_values = [];
      row_model = document.createElement('div');
      row_model.classList.add("row","flex-nowrap","g-0");
      row_pred = document.createElement('div');
      row_pred.classList.add("row","flex-nowrap","g-0");
      for (let i=0;i<4;i++){
        //Model names
        column_model = document.createElement('div');
        column_model.classList.add('col-3','col-pixel-width-100');
        model_name = document.createElement('p')
        model_name.classList.add("text-center");
        model_name.textContent = "-";
        row_models.push(model_name);
        column_model.appendChild(model_name);
        row_model.appendChild(column_model);
        //Prediction part.
        labels = document.createElement('div');
        labels.classList.add('col-2');
        label = document.createElement('div');
        label.innerHTML = '<b>Prediction</b>'
        columns_pred_labels.push(label)
        labels.appendChild(label);
        row_pred.appendChild(labels);
        preds = document.createElement('div');
        preds.classList.add('col-1');
        pred = document.createElement('div');
        pred.innerHTML = '%';
        columns_pred_values.push(pred);
        preds.appendChild(pred);
        row_pred.appendChild(preds);
      }
      e.appendChild(row_model);
      e.appendChild(row_pred);
      e.appendChild(document.createElement('hr'));
    }
    row_models[column].innerHTML = '<b>'+model.name+'</b>';
    for (let i = 0; i < number_of_probabilities_to_show; i++) {
      label = document.createElement('div');
      label.classList.add('small');
      label.innerHTML = model.values[i]['label']
      pred = document.createElement('div');
      pred.classList.add('small');
      pred.innerHTML = (model.values[i]['probability']*100).toFixed(1);
      columns_pred_labels[column].appendChild(label);
      columns_pred_values[column].appendChild(pred);
    }
  });
}
function compare_models(){
  const e = document.getElementById("compare_models");
  let row_e,column_e;
  //For all preds
  probs_v1.forEach((prob_v1, i) => {
    row_e = document.createElement('div');
    row_e.classList.add("row","flex-nowrap","g-0");
    column_e = document.createElement('div');
    column_e.classList.add('col-12');
    let h3_=document.createElement('h3');
    h3_.classList.add("text-center","text-capitalize");
    h3_.innerHTML = prob_v1.label;
    let p_=document.createElement('p');
    p_.classList.add("text-start","small");

    p_.innerHTML ="Saliency maps for the label <b>"+prob_v1.label+" </b> with max class probability <b>"+
    (prob_v1.maxProbability*100).toFixed(1)+"%</b>. The models tested have the mean class probability <b>"+
    (prob_v1.get_average_prob()*100).toFixed(1)+"%</b>"+" and the label is among acc@5 for: <b>"+
    prob_v1.get_number_prob()+"</b> of the models";
    column_e.appendChild(h3_);
    column_e.appendChild(p_);
    row_e.appendChild(column_e);
    let hr_= document.createElement('hr');
    e.appendChild(hr_);
    e.appendChild(row_e);
    add_saliency_maps_for_class(e,prob_v1.classnumber);
    add_chart_for_class(e,prob_v1.classnumber);
    add_raw_chart_for_classnumber(e,prob_v1.classnumber);
    add_dispersion_and_mean_for_class(e,prob_v1.classnumber);
    add_chart_for_mean_and_deviation(e,prob_v1.classnumber);
  });
}
function add_saliency_maps_for_class(element,classnumber){
  let paths_to_saliency_maps = [];
  let class_probabilities = [];
  let model_names = [];
  let mean_values = [];
  let std_values = [];
  let mean_from_pos_value = [];
  let mean_from_neg_value = [];
  let std_from_pos_value = [];
  let std_from_neg_value = [];
  models_v1.forEach((model, i) => {
    let path_to_saliency_map = model.get_saliency_map_for_class(classnumber);
    if (path_to_saliency_map.length > 0){
      paths_to_saliency_maps.push(path_to_saliency_map);
      class_probabilities.push(model.get_class_prob_for_label(classnumber));
      model_names.push(model.get_model_name());
      mean_values.push(model.get_mean_raw_value(classnumber));
      std_values.push(model.get_std_raw_value(classnumber));
      mean_from_pos_value.push(model.get_mean_from_pos_value(classnumber));
      mean_from_neg_value.push(model.get_mean_from_neg_value(classnumber));
      std_from_pos_value.push(model.get_std_from_pos_value(classnumber));
      std_from_neg_value.push(model.get_std_from_neg_value(classnumber));
    };
  });
  let images;
  let column;
  let caps;
  paths_to_saliency_maps.forEach((map_path, i) => {
      column=i%4;
      if (column==0){
        images=[];
        caps=[];
        let row_maps = document.createElement('div');
        row_maps.classList.add("row","flex-nowrap","g-0");
          for (let j = 0; j < 4; j++){
            let column_maps = document.createElement('div');
            column_maps.classList.add("col-3","col-pixel-width-100");
            let img = document.createElement("img");
            img.classList.add("img-thumbnail");
            //img.src = "images/placeholder.png"
            images.push(img);
            let cap = document.createElement("figcaption");
            cap.textContent ="-"
            cap.classList.add("text-left","small")
            caps.push(cap);
            let fig = document.createElement("figure");
            fig.appendChild(img);
            fig.appendChild(cap);
            column_maps.appendChild(fig);
            row_maps.appendChild(column_maps);
          };
      element.appendChild(row_maps);
      }
      //Here only add image so no placeholder is needed.....!!!!!!
      images[column].src = "../"+map_path;
      caps[column].innerHTML = "<i>Model: </i>"+model_names[i]+
      " <br/><i>Class probability:</i> "+ (class_probabilities[i]*100).toFixed(1)+"%"+
      "<br/> <i>Mean value:</i> "+(mean_values[i]).toFixed(2)+
      "<br/> <i>Std value:</i> "+(std_values[i]).toFixed(2)+
      "<br/> <i>Mean of pos values:</i> "+(mean_from_pos_value[i]).toFixed(2)+
      "<br/> <i>Mean of neg values:</i> "+(mean_from_neg_value[i]).toFixed(2)+
      "<br/> <i>Std of pos values:</i> "+(std_from_pos_value[i]).toFixed(2)+
      "<br/> <i>Std of neg values:</i> "+(std_from_neg_value[i]).toFixed(2);
  });
}
function add_chart_for_class(element,classnumber){
  let chart_canvas = document.createElement('canvas');
  let chart_column = document.createElement('div');
  chart_column.classList.add("col-12");
  chart_column.appendChild(chart_canvas);
  let chart_row=document.createElement('div');
  chart_row.classList.add("row","flex-nowrap","g-0","pb-5");
  chart_row.appendChild(chart_column);
  let XAI_values =[]
  const XAI_value = function(mean_values,model_name,target_class_label,colour){
    this.data = mean_values;
    this.label = model_name;
    this.scrap = target_class_label;
    this.borderColor = colour;
    this.backgroundColor =  colour;
  }
  const COLORS = ['#4dc9f6','#f67019','#f53794','#537bc4','#acc236','#166a8f','#00a950','#58595b','#8549ba'];
  models_v1.forEach((model, i) => {
      let saliency_values=model.get_saliency_values(classnumber);
      if (saliency_values.length>0){
        XAI_values.push(new XAI_value(model.get_saliency_values(classnumber),model.get_model_name(),idx_to_label(classnumber),COLORS[i]));
      }
  });
  const labels = [];
  for (var j=1;j<=51;++j){labels.push(j)}
  new Chart(chart_canvas, {
      type: 'line',
      data: {
        labels: labels,

        datasets: XAI_values,
      },
      options: {
        plugins: {
           legend: {
               display: true,
               labels: {
                   font: {
                        size: 14
                    }
               }
           }
       },
        responsive: true,
        title: {
          display: true,
          text: 'Mean value per square ',
        },
        scales: {
        xAxis: {
          title: {
          display: true,
          text: 'Number of squares accumulated over for the label '+idx_to_label(classnumber),
          font: {
               size: 20
           }
        },
          ticks: {
            callback: function(val, index) {
                return index % 5 === 0 ? val : '';
            },
            font: {
                 size: 20
             }
          }
        },
        yAxis:{
          title:{
            display:true,
            text: 'Accumulated mean POSITIVE Occlusion values.',
            font: {
                 size: 20
             }
           },
           suggestedMin:0,
           suggestedMax:5,
             ticks: {
               font: {
                    size: 20
              }
             }
          }

      }
      }
    });
  element.appendChild(chart_row);
}
function add_raw_chart_for_classnumber(element,classnumber){
  let chart_canvas = document.createElement('canvas');
  let chart_column = document.createElement('div');
  chart_column.classList.add("col-12");
  chart_column.appendChild(chart_canvas);
  let chart_row=document.createElement('div');
  chart_row.classList.add("row","flex-nowrap","g-0","pb-5");
  chart_row.appendChild(chart_column);
  let XAI_values =[]
  const XAI_value = function(raw_values,model_name,target_class_label,colour){
    this.data = raw_values;
    this.label = model_name;
    this.scrap = target_class_label;
    this.borderColor = colour;
    this.backgroundColor =  colour;
  }
  const COLORS = ['#4dc9f6','#f67019','#f53794','#537bc4','#acc236','#166a8f','#00a950','#58595b','#8549ba'];
  models_v1.forEach((model, i) => {
      let saliency_values=model.get_raw_values(classnumber);
      if (saliency_values.length>0){
        //console.log("classnumber:",classnumber,"saliency_values",saliency_values)
        XAI_values.push(new XAI_value(model.get_raw_values(classnumber),model.get_model_name(),idx_to_label(classnumber),COLORS[i]));
      }
  });
  const labels = [];
  for (var j=1;j<=51;++j){labels.push(j)}
  new Chart(chart_canvas, {
      type: 'line',
      data: {
        labels: labels,
        datasets: XAI_values,
      },
      options: {
        plugins: {
           legend: {
               display: true,
               labels: {
                   font: {
                        size: 14
                    }
               }
           }
       },
        responsive: true,
        title: {
          display: true,
          text: 'Mean value per square ',
        },
        scales: {
        xAxis: {
          title: {
          display: true,
          text: 'Raw '+idx_to_label(classnumber),
          font: {
               size: 20
           }
        },
          ticks: {
            callback: function(val, index) {
                return index % 5 === 0 ? val : '';
            },
            font: {
                 size: 20
             }
          }
        },
        yAxis:{
          title:{
            display:true,
            text: 'Raw values.',
            font: {
                 size: 20
             }
           },
           suggestedMin:0,
           suggestedMax:5,
             ticks: {
               font: {
                    size: 20
              }
             }
          }

      }
      }
    });
  element.appendChild(chart_row);

}
function add_dispersion_and_mean_for_class(element,classnumber){
  let found_maps = false;
  let mean_disp_title_row = document.createElement('div');
  mean_disp_title_row.classList.add('row','flex-nowrap','g-0');
  let mean_disp_title_column = document.createElement('div');
  mean_disp_title_column.classList.add('col-12','text-center',"h4")
  mean_disp_title_row.appendChild(mean_disp_title_column);
  element.appendChild(mean_disp_title_row);
  //Mean and dispersion row
  // let images = [];
  // let caps = [];
  let row_maps = document.createElement('div');
  row_maps.classList.add("row","flex-nowrap","g-0");
    for (let j = 0; j < 4; j++){
      let column_maps = document.createElement('div');
      column_maps.classList.add("col-3","col-pixel-width-100");
      if (j==1 || j==2){
        let img = document.createElement("img");
        img.classList.add("img-thumbnail");
        //img.src = "images/placeholder.png"

        // images.push(img);
        let cap = document.createElement("figcaption");
        cap.textContent ="-"
        cap.classList.add("text-left","small")
        // caps.push(cap);
        if (j==1){ //mean
            const path = mean_and_dispersion.get_mean_saliency_map_path(classnumber);
            if (path.length>0){
              img.src = path;
              found_maps= true;
              cap.textContent = "Mean saliency map for all models compared. A more green shade indicates more positive correlation between models. (Values (negative) not contributing to the classification ("+idx_to_label(classnumber)+") are removed.)"
            }
        }
        if (j==2){ //dispersion
            const path = mean_and_dispersion.get_dispersion_saliency_map_path(classnumber);
            if (path.length>0){
              img.src = path;
              cap.textContent = "Mean standard deviation map for all models compared. A more red shade means more disagrement. (Values (negative) not contributing to the classification ("+idx_to_label(classnumber)+") are removed.)"
            }
        }
        let fig = document.createElement("figure");
        fig.appendChild(img);
        fig.appendChild(cap);
        column_maps.appendChild(fig);
      }
      row_maps.appendChild(column_maps);
    };
    if(found_maps){
      element.appendChild(row_maps);
      mean_disp_title_column.innerHTML = 'Mean saliency map and dispersion saliency map for <b>'+idx_to_label(classnumber)+'</b>'
    }else{
        mean_disp_title_column.innerHTML = 'No mean saliency map and dispersion saliency maps created for <b>'+idx_to_label(classnumber)+'</b>'
    }

}
function add_chart_for_mean_and_deviation(element,classnumber){
  let found = false;
  let chart_canvas = document.createElement('canvas');
  let chart_column = document.createElement('div');
  chart_column.classList.add("col-12");
  chart_column.appendChild(chart_canvas);
  let chart_row=document.createElement('div');
  chart_row.classList.add("row","flex-nowrap","g-0");
  chart_row.appendChild(chart_column);
  let mean_std_values =[]
  const Mean_std_value = function(values,mean_or_std,colour){
    this.data = values;
    this.label = mean_or_std;
    this.borderColor = colour
    this.backgroundColor =  colour;
  }
  const COLORS = ['#4dc9f6','#f67019','#f53794','#537bc4','#acc236','#166a8f','#00a950','#58595b','#8549ba'];
  const mean_values = mean_and_dispersion.get_mean_values(classnumber);
  if (mean_values.length>0){
    mean_std_values.push(new Mean_std_value(mean_values,"Sorted mean Occlusion value for all squares",COLORS[7]));
    mean_std_values.push(new Mean_std_value(mean_and_dispersion.get_std_values(classnumber),"Sorted standard deviation between models for all squares.",COLORS[8]));
    found=true;
  }
  const labels = [];
  for (var j=0;j<=55;++j){labels.push(j)}
  new Chart(chart_canvas, {
      type: 'line',
      data: {
        labels: labels,
        datasets: mean_std_values,
      },
      options: {
        plugins: {
           legend: {
               display: true,
               labels: {
                   font: {
                        size: 14
                    }
               }
           }
       },
        responsive: true,
        title: {
          display: true,
          text: 'Sorted values for square ',
        },
        scales: {
        xAxis: {
          title: {
          display: true,
          text: idx_to_label(classnumber),
          font: {
               size: 20
           }
        },
          ticks: {
            callback: function(val, index) {
                return index % 5 === 0 ? val : '';
            },
            font: {
                 size: 20
             }
          }
        },
        yAxis:{
          title:{
            display:true,
            text: 'Values',
            font: {
                 size: 20
             }
           },
           suggestedMin:0,
           suggestedMax:4,
             ticks: {
               font: {
                    size: 20
              }
             }
          }

      }
      }
    });
  if(found){
    element.appendChild(chart_row);
  }
}
