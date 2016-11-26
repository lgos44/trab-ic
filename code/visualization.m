d = importdata('data2.csv');

for i = 2:length(d.data(1,:))
    h = figure;
    histogram(d.data(:,i));
    title(d.textdata(i));
    t = d.textdata{i};
    print(t, '-depsc2');
end